# fast-jev-compaction

Conservative context compaction for coding agents. Jev scores completed tool
calls and results, then the library keeps, truncates, or removes those pairs
without rewriting user or assistant text.

It uses OpenRouter by default:

- endpoint: `https://openrouter.ai/api/v1/chat/completions`
- model: `~typesafe/jev-latest`
- credential: `OPENROUTER_API_KEY`

The TypeSafe System One endpoint remains available only with
`provider: 'typesafe'`; it is never an automatic fallback.

## Install

```sh
npm install fast-jev-compaction
export OPENROUTER_API_KEY=...
```

```ts
import { compactMessages, reductionRatio, type Message } from 'fast-jev-compaction';

const transcript: Message[] = [
  { role: 'user', text: 'Fix the failing test. Do not edit generated files.', toolUses: [] },
  {
    role: 'assistant',
    text: '',
    toolUses: [{ tool_use_id: 'read-1', tool: 'Read', input: { file_path: 'src/a.ts' } }],
  },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'read-1', text: '…file…' }] },
];

const result = await compactMessages(transcript, { preserveRecentMessages: 4 });
console.log(result.messages, result.decisions, reductionRatio(result));
```

OpenRouter provides Jev through its OpenAI-compatible API. The library wraps
the conversation state and Jev questions in a chat request, requires a strict
JSON-schema response, and validates every probability before changing a
transcript.

## Safety policy

Compaction is fail-open: an invalid response, request timeout, ambiguous tool
pairing, inadequate classifier state, or a decision to drop every scored pair
throws. An agent adapter should then keep its native history or use its native
compactor.

By default, failed tool results and `Edit`, `Write`, `NotebookEdit`, and
`apply_patch` pairs are preserved. Add project-specific tools with
`neverDeleteTools`. Jev sees a short head-and-tail preview of each completed
result so it can score content rather than only tool names and output lengths.
This means selected conversation text is sent to OpenRouter; do not use the
default transport for transcripts that must not leave the machine.

| Option | Default | Purpose |
| --- | ---: | --- |
| `keepCallThreshold` | `0.5` | Keep a record that a tool call happened. |
| `keepResultThreshold` | `0.25` | Keep the full result; separate because Jev calibrates result scores lower. |
| `preserveRecentMessages` | `6` | Pin the newest messages, plus the first message. |
| `neverDeleteTools` | edits and writes | Tool names that are always preserved. |
| `protectErrors` | `true` | Preserve failed tool calls and results. |
| `resultPreviewChars` | `160` | Characters from each result end shown to Jev. |
| `truncateHeadChars` | `300` | Result characters retained when only the call is kept. |
| `maxStateTokens` | `25000` | Estimated classifier-state ceiling. |
| `maxRequestTokens` | `30000` | Estimated state-plus-question ceiling. |
| `maxConcurrentRequests` | `4` | Per-compaction concurrency limit. |
| `timeoutMs` | `15000` | Client deadline, including response body. |

`keepThreshold` is retained for callers that intentionally want one legacy
threshold for both decisions. Values outside `[0, 1]` are rejected.

## Agent integrations

### Codex

This repository is a valid Codex plugin with the `jev-context` skill in
[`skills/jev-context`](skills/jev-context/SKILL.md). It gives Codex the
conservative retention rules and the OpenRouter configuration when an adapter
has a transcript to compact.

Current public Codex hooks can observe and stop compaction, but cannot replace
Codex's compacted transcript. The plugin therefore does **not** claim to
override Codex's native compactor. Use the npm library for a transcript-aware
Codex integration, or use the skill to guide a manual retention workflow.

### Hermes Agent

[`integrations/hermes/context_engine/fast_jev_compaction`](integrations/hermes/context_engine/fast_jev_compaction)
is a standard-library Hermes context engine. Copy that directory into the
Hermes `plugins/context_engine/` directory, then select it:

```yaml
# config.yaml
context:
  engine: fast_jev_compaction
```

Set `OPENROUTER_API_KEY` before starting Hermes. The engine is fail-open,
understands OpenAI tool-call and tool-result messages, protects errors and
edits, and returns the original message list when its evidence is incomplete.

### Claude Code

The historic Claude function-hook adapter remains in `hooks/` for compatible
early-access builds. Public Claude Code releases have not provided the
transcript-replacement events it requires, so it is not presented as a
supported installation path. Its adapter follows the same OpenRouter defaults,
timeout, locking, and fallback rules when used with a compatible host.

## Development

The published library supports Node.js 18 or later. Running this repository's
Vitest 4 test suite requires Node.js 20 or later.

```sh
npm ci
npm run typecheck
npm test
npm run build
python3 -m unittest integrations/hermes/tests/test_context_engine.py
```

`prepack` builds `dist/`, so a clean npm package contains the exported JavaScript
and type declarations. The unit tests use fake OpenRouter responses and never
send a transcript to the network.
