import { noulAnswer } from './request.js';
import { collectToolCalls, estimateTokens, fitState } from './state.js';
import type {
  CallAnswer,
  CallDecision,
  CompactOptions,
  CompactResult,
  CompactionState,
  JevAsker,
  JevQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepCallThreshold: 0.5,
  keepResultThreshold: 0.25,
  preserveRecentMessages: 6,
  maxStateTokens: 25_000,
  maxRequestTokens: 30_000,
  truncateHeadChars: 300,
  resultPreviewChars: 160,
  neverDeleteTools: ['Edit', 'Write', 'NotebookEdit', 'apply_patch'],
  protectErrors: true,
  failOnAllCandidatesDropped: true,
  maxConcurrentRequests: 4,
};

/** Tokens for the chat envelope, JSON schema and model instructions around state/questions. */
const REQUEST_OVERHEAD_TOKENS = 160;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function probability(value: number | undefined, fallback: number, name: string): number {
  const resolved = finite(value, fallback);
  if (resolved < 0 || resolved > 1) {
    throw new Error(`${name} must be between 0 and 1`);
  }
  return resolved;
}

function boolean(value: boolean | undefined, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function nonNegativeInteger(value: number | undefined, fallback: number): number {
  return Math.max(0, Math.floor(finite(value, fallback)));
}

/** Resolves public options and rejects thresholds that could make certain keeps delete data. */
export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  const legacyThreshold = options.keepThreshold;
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepCallThreshold: probability(
      options.keepCallThreshold ?? legacyThreshold,
      DEFAULT_OPTIONS.keepCallThreshold,
      'keepCallThreshold',
    ),
    keepResultThreshold: probability(
      options.keepResultThreshold ?? legacyThreshold,
      DEFAULT_OPTIONS.keepResultThreshold,
      'keepResultThreshold',
    ),
    preserveRecentMessages: nonNegativeInteger(
      options.preserveRecentMessages,
      DEFAULT_OPTIONS.preserveRecentMessages,
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    truncateHeadChars: nonNegativeInteger(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars),
    resultPreviewChars: nonNegativeInteger(
      options.resultPreviewChars,
      DEFAULT_OPTIONS.resultPreviewChars,
    ),
    neverDeleteTools: options.neverDeleteTools
      ? [...options.neverDeleteTools]
      : DEFAULT_OPTIONS.neverDeleteTools,
    protectErrors: boolean(options.protectErrors, DEFAULT_OPTIONS.protectErrors),
    failOnAllCandidatesDropped: boolean(
      options.failOnAllCandidatesDropped,
      DEFAULT_OPTIONS.failOnAllCandidatesDropped,
    ),
    maxConcurrentRequests: Math.max(
      1,
      nonNegativeInteger(options.maxConcurrentRequests, DEFAULT_OPTIONS.maxConcurrentRequests),
    ),
  };
}

/** The two `noul` questions asked about one call: retaining a record, and retaining full output. */
export function questionsFor(call: ToolCall): JevQuestions {
  return {
    [`call_${call.id}`]: {
      type: 'noul',
      instructions: `Tool call ${call.id} (${call.tool}) carries information the current task still depends on: a file or command being worked with, a decision, a constraint, or a change that was made.`,
    },
    [`result_${call.id}`]: {
      type: 'noul',
      instructions: `The output of tool call ${call.id} (${call.tool}, ${call.resultChars} chars) contains information the assistant needs again to continue correctly, such as an error, value, file content, constraint, or point-in-time fact.`,
    },
  };
}

/**
 * Splits candidate calls into batches whose questions, together with the
 * complete state, fit one request.
 */
export function batchCalls(
  calls: readonly ToolCall[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens'>,
): ToolCall[][] {
  const budget = options.maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  const batches: ToolCall[][] = [];
  let current: ToolCall[] = [];
  let currentTokens = 0;
  for (const call of calls) {
    const tokens = estimateTokens(JSON.stringify(questionsFor(call)));
    if (current.length > 0 && currentTokens + tokens > budget) {
      batches.push(current);
      current = [];
      currentTokens = 0;
    }
    if (current.length === 0 && tokens > budget) {
      throw new Error(
        `state leaves no room for questions (~${stateTokens} of ${options.maxRequestTokens} tokens)`,
      );
    }
    current.push(call);
    currentTokens += tokens;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

type DecisionThresholds = Pick<ResolvedCompactOptions, 'keepCallThreshold' | 'keepResultThreshold'>;

export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned' | 'protected'>,
  answer: CallAnswer,
  options: DecisionThresholds,
): CallDecision {
  const base = { id: call.id, tool: call.tool, ...answer };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (call.protected) return { ...base, action: 'keep', reason: 'protected' };
  if (answer.keepResult >= options.keepResultThreshold) {
    return { ...base, action: 'keep', reason: 'kept' };
  }
  if (answer.keepCall >= options.keepCallThreshold) {
    return { ...base, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

async function askBatch(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly ToolCall[],
): Promise<Map<string, CallAnswer>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map(questionsFor));
  const { answers } = await asker.ask(state, questions);
  return new Map(
    batch.map((call) => [
      call.id,
      {
        keepCall: noulAnswer(answers, `call_${call.id}`),
        keepResult: noulAnswer(answers, `result_${call.id}`),
      },
    ]),
  );
}

/** Runs a bounded queue and waits for already-started requests before rejecting. */
async function askBatches(
  asker: JevAsker,
  state: CompactionState,
  batches: readonly ToolCall[][],
  maxConcurrentRequests: number,
): Promise<Map<string, CallAnswer>[]> {
  const answers: Map<string, CallAnswer>[] = [];
  let next = 0;
  let failure: unknown;
  const worker = async (): Promise<void> => {
    while (failure === undefined) {
      const index = next;
      next += 1;
      if (index >= batches.length) return;
      try {
        answers[index] = await askBatch(asker, state, batches[index]!);
      } catch (error) {
        failure = error;
      }
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(maxConcurrentRequests, batches.length) }, () => worker()),
  );
  if (failure !== undefined) throw failure;
  return answers;
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-jev-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/**
 * Rebuilds the conversation from decisions. A dropped call disappears with
 * its result; a dropped result keeps a bounded head and note. Protected and
 * pinned records are never mutated even if a malformed caller supplies a drop.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && !call.pinned && !call.protected && decision.action !== 'keep') {
      actions.set(call.tool_use_id, decision.action);
    }
  }
  const kept: Message[] = [];
  for (const message of messages) {
    const touched =
      message.toolUses.some((tool) => actions.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => actions.has(result.tool_use_id));
    if (!touched) {
      kept.push(message);
      continue;
    }
    const toolUses = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        if (actions.get(tool.tool_use_id) !== 'drop_result') return tool;
        const text = truncatedResultText(tool.text ?? '', tool.isError ?? false, headChars);
        if ((tool.text ?? '') === text) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: tool.input,
          text,
        };
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        if (actions.get(result.tool_use_id) !== 'drop_result') return result;
        const text = truncatedResultText(result.text, result.isError ?? false, headChars);
        return text === result.text ? result : { tool_use_id: result.tool_use_id, text, isError: result.isError };
      });
    if (
      !message.toolUses.some((tool) => actions.get(tool.tool_use_id) === 'drop_call') &&
      !(message.toolResults ?? []).some((result) => actions.get(result.tool_use_id) === 'drop_call') &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every((result, index) => result === message.toolResults?.[index])
    ) {
      kept.push(message);
      continue;
    }
    if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

function largestQuestionTokens(calls: readonly ToolCall[]): number {
  return Math.max(0, ...calls.map((call) => estimateTokens(JSON.stringify(questionsFor(call)))));
}

function stateShowsAllCandidates(state: CompactionState, candidates: readonly ToolCall[]): boolean {
  const visible = new Set<string>();
  for (const entry of state.history) {
    for (const tool of entry.tool_calls ?? []) {
      if (typeof tool === 'string') visible.add(tool.split(' ', 1)[0] ?? '');
      else visible.add(tool.id);
    }
  }
  return candidates.every((call) => visible.has(call.id));
}

/**
 * Compacts a transcript by asking Jev about completed, unprotected tool pairs.
 * It fails closed on malformed data, missing classifier evidence, or a
 * degenerate all-drop result so callers can retain their native history.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages, resolved);
  const candidates = calls.filter((call) => !call.pinned && !call.protected);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let fitted: { tokens: number; stage: string } = { tokens: 0, stage: '' };
  let batches: ToolCall[][] = [];
  const answers = new Map<string, CallAnswer>();
  if (candidates.length > 0) {
    const availableStateTokens = Math.min(
      resolved.maxStateTokens,
      resolved.maxRequestTokens - REQUEST_OVERHEAD_TOKENS - largestQuestionTokens(candidates),
    );
    if (availableStateTokens < 1) {
      throw new Error(`request budget ${resolved.maxRequestTokens} leaves no room for Jev state`);
    }
    const state = fitState(messages, calls, { ...resolved, maxStateTokens: availableStateTokens });
    if (!stateShowsAllCandidates(state.state, candidates)) {
      throw new Error('classifier state no longer contains every scored tool call');
    }
    fitted = state;
    batches = batchCalls(candidates, state.tokens, resolved);
    const answered = await askBatches(
      asker,
      state.state,
      batches,
      resolved.maxConcurrentRequests,
    );
    for (const map of answered) for (const [id, answer] of map) answers.set(id, answer);
  }

  const decisions = calls.map((call) =>
    decideCall(call, answers.get(call.id) ?? { keepCall: 1, keepResult: 1 }, resolved),
  );
  const scored = decisions.filter((decision) => decision.reason !== 'pinned' && decision.reason !== 'protected');
  if (
    resolved.failOnAllCandidatesDropped &&
    scored.length > 0 &&
    scored.every((decision) => decision.action === 'drop_call')
  ) {
    throw new Error('Jev would drop every scored tool pair; retaining the original transcript');
  }
  const kept = applyDecisions(messages, decisions, calls, resolved.truncateHeadChars);
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      protected: count(decisions, 'protected'),
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests: batches.length,
      ms: Date.now() - started,
    },
  };
}
