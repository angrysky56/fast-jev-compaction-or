import type { JevAnswer, JevQuestions, JevResponse, JevState } from './types.js';

/** The public OpenRouter endpoint used by default. */
export const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
/** Retained for users who explicitly opt in to TypeSafe's private System One API. */
export const SYSTEM_ONE_URL = 'https://api.typesafe.ai/v1/systemone';
export const DEFAULT_MODEL = '~typesafe/jev-latest';

export type JevProvider = 'openrouter' | 'typesafe';

export interface JevRequest {
  url: string;
  method: 'POST';
  headers: Record<string, string>;
  body: string;
}

const OPENROUTER_SYSTEM_PROMPT = [
  'You are Jev, a fast structured relevance classifier for a coding-agent transcript.',
  'Return only the requested JSON object. For each question, set noul to the probability from 0 to 1 that the described information should remain available to the agent.',
  'Use the supplied goal and transcript evidence. Prefer keeping an item when losing it could cause a repeated action, loss of a constraint, or loss of a non-reproducible fact.',
].join(' ');

function answerSchema(questions: JevQuestions): object {
  const properties = Object.fromEntries(
    Object.keys(questions).map((name) => [
      name,
      {
        type: 'object',
        properties: {
          type: { const: 'noul' },
          noul: { type: 'number', minimum: 0, maximum: 1 },
        },
        required: ['type', 'noul'],
        additionalProperties: false,
      },
    ]),
  );
  return {
    type: 'object',
    properties: {
      answers: {
        type: 'object',
        properties,
        required: Object.keys(questions),
        additionalProperties: false,
      },
    },
    required: ['answers'],
    additionalProperties: false,
  };
}

/** Builds either the OpenRouter chat request or an explicitly selected native TypeSafe request. */
export function buildJevRequest(
  params: {
    apiKey: string;
    model?: string;
    baseUrl?: string;
    provider?: JevProvider;
  },
  state: JevState,
  questions: JevQuestions,
): JevRequest {
  const provider = params.provider ?? 'openrouter';
  const model = params.model ?? DEFAULT_MODEL;
  if (provider === 'typesafe') {
    return {
      url: params.baseUrl ?? SYSTEM_ONE_URL,
      method: 'POST',
      headers: {
        authorization: `Bearer ${params.apiKey}`,
        'content-type': 'application/json',
      },
      body: JSON.stringify({ model, state, questions }),
    };
  }
  return {
    url: params.baseUrl ?? OPENROUTER_URL,
    method: 'POST',
    headers: {
      authorization: `Bearer ${params.apiKey}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: OPENROUTER_SYSTEM_PROMPT },
        { role: 'user', content: JSON.stringify({ state, questions }) },
      ],
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: 'jev_decisions',
          strict: true,
          schema: answerSchema(questions),
        },
      },
      max_tokens: Math.max(128, Object.keys(questions).length * 12),
      temperature: 0,
    }),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function responseWithAnswers(value: unknown): JevResponse {
  if (!isRecord(value) || !isRecord(value.answers)) {
    throw new Error('Jev response is missing an answers object');
  }
  return value as JevResponse;
}

/** Validates the native System One response envelope. */
export function parseJevResponse(status: number, ok: boolean, text: string): JevResponse {
  if (!ok) {
    throw new Error(`Jev request failed (${status}): ${text.slice(0, 200)}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error('Jev returned malformed JSON');
  }
  return responseWithAnswers(parsed);
}

function openRouterContent(parsed: Record<string, unknown>): string {
  const choices = parsed.choices;
  if (!Array.isArray(choices) || !isRecord(choices[0]) || !isRecord(choices[0].message)) {
    throw new Error('OpenRouter response is missing a message choice');
  }
  const content = choices[0].message.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter(isRecord)
      .map((part) => (typeof part.text === 'string' ? part.text : ''))
      .join('');
  }
  throw new Error('OpenRouter response has no JSON content');
}

/** Parses OpenRouter's OpenAI-compatible response into Jev's native answer shape. */
export function parseOpenRouterResponse(status: number, ok: boolean, text: string): JevResponse {
  if (!ok) {
    throw new Error(`OpenRouter Jev request failed (${status}): ${text.slice(0, 200)}`);
  }
  let envelope: unknown;
  try {
    envelope = JSON.parse(text);
  } catch {
    throw new Error('OpenRouter returned malformed JSON');
  }
  if (!isRecord(envelope)) throw new Error('OpenRouter returned an invalid response envelope');
  let content: unknown;
  try {
    content = JSON.parse(openRouterContent(envelope));
  } catch (error) {
    if (error instanceof Error && error.message.startsWith('OpenRouter response')) throw error;
    throw new Error('OpenRouter Jev response did not contain valid JSON');
  }
  const response = responseWithAnswers(content);
  if (typeof envelope.model === 'string') response.model = envelope.model;
  if (isRecord(envelope.usage)) {
    response.usage = {
      input_tokens:
        typeof envelope.usage.prompt_tokens === 'number' ? envelope.usage.prompt_tokens : undefined,
      output_tokens:
        typeof envelope.usage.completion_tokens === 'number'
          ? envelope.usage.completion_tokens
          : undefined,
    };
  }
  return response;
}

/** The `noul` probability of one answer; rejects malformed or unsafe values before deletion. */
export function noulAnswer(answers: Record<string, JevAnswer>, name: string): number {
  if (!Object.prototype.hasOwnProperty.call(answers, name)) {
    throw new Error(`Invalid Jev answer for ${name}`);
  }
  const answer = answers[name];
  if (
    !isRecord(answer) ||
    (Object.prototype.hasOwnProperty.call(answer, 'type') && answer.type !== 'noul') ||
    !Object.prototype.hasOwnProperty.call(answer, 'noul') ||
    typeof answer.noul !== 'number' ||
    !Number.isFinite(answer.noul) ||
    answer.noul < 0 ||
    answer.noul > 1
  ) {
    throw new Error(`Invalid Jev answer for ${name}`);
  }
  return answer.noul;
}
