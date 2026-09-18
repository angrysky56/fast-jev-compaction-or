import {
  buildJevRequest,
  parseJevResponse,
  parseOpenRouterResponse,
  type JevProvider,
} from './request.js';
import type { JevAsker, JevQuestions, JevResponse, JevState } from './types.js';

export interface JevClientOptions {
  /** OpenRouter is the default. Use `typesafe` only with an explicit private-API key. */
  provider?: JevProvider;
  /** Defaults to `OPENROUTER_API_KEY`, or `TYPESAFE_API_KEY` for `provider: 'typesafe'`. */
  apiKey?: string;
  /** Defaults to `~typesafe/jev-latest`. */
  model?: string;
  /** Overrides the selected provider's endpoint. */
  baseUrl?: string;
  /** Defaults to 15 seconds, including reading the response body. */
  timeoutMs?: number;
  /** Cancels the request when the caller no longer needs a compaction result. */
  signal?: AbortSignal;
  /** Defaults to the global `fetch`. */
  fetch?: typeof fetch;
}

function positiveFinite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;
}

function requestSignal(signal: AbortSignal | undefined, timeoutMs: number): {
  signal: AbortSignal;
  dispose: () => void;
} {
  const controller = new AbortController();
  const abortFromCaller = (): void => controller.abort(signal?.reason);
  if (signal?.aborted) abortFromCaller();
  else signal?.addEventListener('abort', abortFromCaller, { once: true });
  const timer = setTimeout(() => controller.abort(new Error('Jev request timed out')), timeoutMs);
  return {
    signal: controller.signal,
    dispose: () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abortFromCaller);
    },
  };
}

/** Asks Jev over OpenRouter by default, with deadline and caller cancellation support. */
export class JevClient implements JevAsker {
  private readonly provider: JevProvider;
  private readonly apiKey: string;
  private readonly model: string | undefined;
  private readonly baseUrl: string | undefined;
  private readonly timeoutMs: number;
  private readonly signal: AbortSignal | undefined;
  private readonly fetcher: typeof fetch;

  constructor(options: JevClientOptions = {}) {
    this.provider = options.provider ?? 'openrouter';
    this.apiKey =
      options.apiKey ??
      (this.provider === 'typesafe'
        ? process.env.TYPESAFE_API_KEY ?? ''
        : process.env.OPENROUTER_API_KEY ?? '');
    this.model = options.model;
    this.baseUrl = options.baseUrl;
    this.timeoutMs = positiveFinite(options.timeoutMs, 15_000);
    this.signal = options.signal;
    this.fetcher = options.fetch ?? fetch;
  }

  async ask(state: JevState, questions: JevQuestions): Promise<JevResponse> {
    if (!this.apiKey) {
      throw new Error(
        this.provider === 'typesafe'
          ? 'TYPESAFE_API_KEY is not configured'
          : 'OPENROUTER_API_KEY is not configured',
      );
    }
    const request = buildJevRequest(
      {
        apiKey: this.apiKey,
        model: this.model,
        baseUrl: this.baseUrl,
        provider: this.provider,
      },
      state,
      questions,
    );
    const deadline = requestSignal(this.signal, this.timeoutMs);
    try {
      const response = await this.fetcher(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
        signal: deadline.signal,
      });
      const text = await response.text();
      return this.provider === 'typesafe'
        ? parseJevResponse(response.status, response.ok, text)
        : parseOpenRouterResponse(response.status, response.ok, text);
    } catch (error) {
      if (deadline.signal.aborted) {
        const reason = deadline.signal.reason;
        throw new Error(
          reason instanceof Error && reason.message ? reason.message : 'Jev request was cancelled',
          { cause: error },
        );
      }
      throw error;
    } finally {
      deadline.dispose();
    }
  }
}
