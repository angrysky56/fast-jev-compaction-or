import { describe, expect, it } from 'vitest';
import {
  applyDecisions,
  batchCalls,
  buildJevRequest,
  collectToolCalls,
  compact,
  decideCall,
  fitState,
  JevClient,
  noulAnswer,
  parseJevResponse,
  parseOpenRouterResponse,
  resolveOptions,
  type JevAsker,
  type JevQuestions,
  type Message,
  type ToolCall,
} from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text = ''): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

function transcript(): Message[] {
  return [
    message('user', 'Fix the failing test. Never edit generated files.'),
    call('read-1', 'Read', { file_path: 'src/a.ts' }, 'export const a = 1;\n'.repeat(30)),
    result('read-1', 'export const a = 1;\n'.repeat(30)),
    call('edit-1', 'Edit', { file_path: 'src/a.ts' }, 'updated'),
    result('edit-1', 'updated'),
    call('bash-1', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('bash-1', 'FAIL a.test.ts: expected 2 to be 3', true),
    message('assistant', 'I will fix src/a.ts.'),
  ];
}

function fakeJev(answer: (name: string) => number, seen: JevQuestions[] = []): JevAsker {
  return {
    async ask(_state, questions) {
      seen.push(questions);
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: answer(key) }]),
        ),
      };
    },
  };
}

describe('options and decisions', () => {
  it('uses calibrated, separate defaults and supports the legacy threshold explicitly', () => {
    expect(resolveOptions()).toMatchObject({
      keepCallThreshold: 0.5,
      keepResultThreshold: 0.25,
      protectErrors: true,
      maxConcurrentRequests: 4,
    });
    expect(resolveOptions({ keepThreshold: 0.4 })).toMatchObject({
      keepCallThreshold: 0.4,
      keepResultThreshold: 0.4,
    });
    expect(() => resolveOptions({ keepResultThreshold: 1.01 })).toThrow(/between 0 and 1/);
  });

  it('does not compare result and call probabilities on the same scale', () => {
    const options = resolveOptions();
    expect(
      decideCall({ id: 't1', tool: 'Read', pinned: false }, { keepCall: 0.7, keepResult: 0.3 }, options),
    ).toMatchObject({ action: 'keep', reason: 'kept' });
    expect(
      decideCall({ id: 't2', tool: 'Read', pinned: false }, { keepCall: 0.7, keepResult: 0.2 }, options),
    ).toMatchObject({ action: 'drop_result' });
    expect(
      decideCall({ id: 't3', tool: 'Read', pinned: false, protected: true }, { keepCall: 0, keepResult: 0 }, options),
    ).toMatchObject({ action: 'keep', reason: 'protected' });
  });
});

describe('transcript safety and state', () => {
  it('protects edits and errors, carries result previews, and keeps pending calls visible', () => {
    const messages = [...transcript(), call('pending-1', 'Read', { file_path: 'src/next.ts' })];
    const options = resolveOptions({ preserveRecentMessages: 0 });
    const calls = collectToolCalls(messages, 0, options);
    expect(calls.map((entry) => [entry.tool, entry.protected])).toEqual([
      ['Read', false],
      ['Edit', true],
      ['Bash', true],
    ]);
    const state = fitState(messages, calls, { ...options, maxStateTokens: 20_000 });
    const first = state.state.history.find((entry) => entry.tool_calls)?.tool_calls?.[0];
    expect(JSON.stringify(first)).toContain('preview=');
    expect(JSON.stringify(first)).toContain('export const a = 1');
    expect(state.state.history.at(-1)?.pending_calls).toEqual([
      { tool: 'Read', input: '{"file_path":"src/next.ts"}' },
    ]);
  });

  it('rejects ambiguous pairs and refuses to apply a destructive decision to a pinned call', () => {
    const duplicate = [message('user', 'start'), call('x', 'Read', {}), result('x', 'a'), call('x', 'Read', {})];
    expect(() => collectToolCalls(duplicate, 0)).toThrow(/duplicate tool call/);
    const messages = [message('user', 'start'), call('x', 'Read', {}, 'old'), result('x', 'old')];
    const calls = collectToolCalls(messages, 2, resolveOptions({ protectErrors: false, neverDeleteTools: [] }));
    const decisions = [
      { id: 't1', tool: 'Read', keepCall: 0, keepResult: 0, action: 'drop_call' as const, reason: 'call_dropped' as const },
    ];
    expect(applyDecisions(messages, decisions, calls, 0)).toEqual(messages);
  });
});

describe('request budgeting and bounded work', () => {
  const calls: ToolCall[] = Array.from({ length: 8 }, (_, index) => ({
    id: `t${index + 1}`,
    tool_use_id: `tool-${index + 1}`,
    tool: 'Read',
    input: {},
    callIndex: index + 1,
    resultIndex: index + 2,
    resultChars: 20,
    isError: false,
    pinned: false,
  }));

  it('splits batches when a request has little question capacity', () => {
    const batches = batchCalls(calls, 29_000, resolveOptions());
    expect(batches.length).toBeGreaterThan(1);
    expect(batches.flat().map((entry) => entry.id)).toEqual(calls.map((entry) => entry.id));
  });

  it('limits concurrently dispatched asks', async () => {
    const messages: Message[] = [message('user', 'start')];
    for (let index = 0; index < 6; index += 1) {
      messages.push(call(`r${index}`, 'Read', { file_path: `${index}.ts` }), result(`r${index}`, 'x'));
    }
    let active = 0;
    let peak = 0;
    const asker: JevAsker = {
      async ask(_state, questions) {
        active += 1;
        peak = Math.max(peak, active);
        await new Promise((resolve) => setTimeout(resolve, 5));
        active -= 1;
        return { answers: Object.fromEntries(Object.keys(questions).map((name) => [name, { noul: 0.8 }])) };
      },
    };
    const output = await compact(messages, asker, {
      preserveRecentMessages: 0,
      maxRequestTokens: 550,
      maxConcurrentRequests: 2,
    });
    expect(output.stats.requests).toBeGreaterThan(2);
    expect(peak).toBeLessThanOrEqual(2);
  });
});

describe('compaction behavior', () => {
  it('fails closed when Jev would delete every scored pair', async () => {
    await expect(
      compact(transcript(), fakeJev(() => 0), { preserveRecentMessages: 0 }),
    ).rejects.toThrow(/would drop every scored tool pair/);
  });

  it('keeps protected records while modifying an unprotected result', async () => {
    const output = await compact(
      transcript(),
      fakeJev((name) => (name.startsWith('call_') ? 0.8 : 0.1)),
      { preserveRecentMessages: 0 },
    );
    expect(output.decisions.map((entry) => entry.reason)).toEqual([
      'result_dropped',
      'protected',
      'protected',
    ]);
    expect(output.messages.some((entry) => entry.toolUses.some((tool) => tool.tool === 'Edit'))).toBe(true);
    expect(output.stats.protected).toBe(2);
  });
});

describe('OpenRouter transport', () => {
  const questions = { q: { type: 'noul' as const, instructions: 'keep it?' } };

  it('defaults to OpenRouter and sends a strict JSON-schema chat request', () => {
    const request = buildJevRequest({ apiKey: 'key' }, { state: true }, questions);
    const body = JSON.parse(request.body) as Record<string, unknown>;
    expect(request.url).toBe('https://openrouter.ai/api/v1/chat/completions');
    expect(body.model).toBe('~typesafe/jev-latest');
    expect(body.messages).toHaveLength(2);
    expect((body.response_format as { type: string }).type).toBe('json_schema');
  });

  it('keeps TypeSafe available only as an explicit compatibility transport', () => {
    const request = buildJevRequest({ apiKey: 'key', provider: 'typesafe' }, 'state', questions);
    expect(request.url).toBe('https://api.typesafe.ai/v1/systemone');
    expect(JSON.parse(request.body)).toMatchObject({ model: '~typesafe/jev-latest', state: 'state', questions });
  });

  it('validates native and OpenRouter answers before they can affect a transcript', () => {
    expect(() => parseJevResponse(200, true, '{"answers":[]}')).toThrow(/answers object/);
    expect(() => noulAnswer({ q: { type: 'noul', noul: 1.2 } }, 'q')).toThrow(/Invalid Jev answer/);
    expect(
      parseOpenRouterResponse(
        200,
        true,
        JSON.stringify({ choices: [{ message: { content: '{"answers":{"q":{"type":"noul","noul":0.4}}}' } }] }),
      ).answers.q,
    ).toEqual({ type: 'noul', noul: 0.4 });
  });

  it('uses OPENROUTER_API_KEY and parses an OpenRouter response', async () => {
    const bodies: string[] = [];
    const client = new JevClient({
      apiKey: 'key',
      fetch: (async (_url, init) => {
        bodies.push(String(init?.body));
        return new Response(
          JSON.stringify({ choices: [{ message: { content: '{"answers":{"q":{"type":"noul","noul":0.4}}}' } }] }),
        );
      }) as typeof fetch,
    });
    expect((await client.ask('state', questions)).answers.q).toMatchObject({ noul: 0.4 });
    expect(JSON.parse(bodies[0]!).model).toBe('~typesafe/jev-latest');
    await expect(new JevClient({ apiKey: '' }).ask('state', questions)).rejects.toThrow(/OPENROUTER_API_KEY/);
  });
});
