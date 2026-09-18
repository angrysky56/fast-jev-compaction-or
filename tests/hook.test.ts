import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLogLines,
  resolveHookConfig,
  toSessionMessages,
} from '../hooks/fast-jev.ts';
import { applyDecisions, collectToolCalls, decideCall, resolveOptions, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input: { file_path: 'src/a.ts' }, text }],
    handle: `call-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `result-${id}` });
}

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the test.', { handle: 'start' }),
    call('read', 'Read', 'x'.repeat(800)),
    result('read', 'x'.repeat(800)),
    call('edit', 'Edit', 'updated'),
    result('edit', 'updated'),
    message('assistant', 'Now editing.', { handle: 'end' }),
  ];
}

function openRouterFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    const body = JSON.parse(init?.body ?? '{}') as { messages: Array<{ content: string }> };
    bodies.push(init?.body ?? '');
    const input = JSON.parse(body.messages[1]!.content) as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(input.questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return {
      status: 200,
      ok: true,
      text: JSON.stringify({ choices: [{ message: { content: JSON.stringify({ answers }) } }] }),
    };
  };
}

describe('legacy Claude adapter configuration', () => {
  it('defaults to OpenRouter and permits TypeSafe only when explicitly selected', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      model: '~typesafe/jev-latest',
      provider: 'openrouter',
      timeoutMs: 15_000,
    });
    expect(resolveHookConfig({ provider: 'typesafe', model: 'jev-legacy' })).toMatchObject({
      provider: 'typesafe',
      model: 'jev-legacy',
    });
  });

  it('keeps original host handles for untouched session messages', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0, resolveOptions({ protectErrors: false, neverDeleteTools: [] }));
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.8, keepResult: 0.1 }, resolveOptions()),
      decideCall(calls[1]!, { keepCall: 0, keepResult: 0 }, resolveOptions()),
    ];
    const output = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(output[0]).toBe(messages[0]);
    expect(output[1]?.handle).toBeUndefined();
    expect(output.some((entry) => entry.handle === 'call-edit')).toBe(false);
  });
});

describe('legacy Claude adapter execution', () => {
  it('sends compact instructions with the Jev goal and keeps protected edits', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 0 }), apiKey: 'key' };
    const output = await compactSession(
      transcript(),
      config,
      openRouterFetch((name) => (name.startsWith('call_') ? 0.8 : 0.1), bodies),
      'Prioritize the failing assertion.',
    );
    const request = JSON.parse(bodies[0]!) as { messages: Array<{ content: string }> };
    const input = JSON.parse(request.messages[1]!.content) as { state: { goal: string } };
    expect(input.state.goal).toContain('Prioritize the failing assertion.');
    expect(output.result.stats.protected).toBe(1);
    expect(output.messages.some((entry) => entry.toolUses.some((tool) => tool.tool === 'Edit'))).toBe(true);
    expect(decisionLogLines(output.result, 80).every((line) => line.length <= 80)).toBe(true);
  });

  it('reports the correct missing-key name', async () => {
    await expect(
      compactSession(transcript(), resolveHookConfig({ preserveRecentMessages: 0 }), openRouterFetch(() => 0)),
    ).rejects.toThrow(/OPENROUTER_API_KEY/);
  });
});
