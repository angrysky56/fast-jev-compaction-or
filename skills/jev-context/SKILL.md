---
name: jev-context
description: Use OpenRouter-backed Jev scoring to preserve useful tool evidence when adapting or compacting an agent transcript.
---

# Jev Context Compaction

Use this skill when an agent integration supplies a transcript and needs to
reduce tool-call context without discarding useful evidence. The default model
is `~typesafe/jev-latest` through OpenRouter and uses `OPENROUTER_API_KEY`.

Keep user and assistant text in order. Only completed tool pairs may be
considered for compaction. Preserve failed commands, edits, writes, and any
tool the caller marks non-deletable. Treat an invalid response, an incomplete
classifier state, or an all-drop decision as a failed compaction and keep the
host's original context.

Codex's public hook API does not replace the transcript produced by its native
compactor. Use the library for transcript-aware adapters or a retention report;
do not claim that this skill changes Codex's built-in compaction. Hermes Agent
can use the bundled context-engine integration, which owns its message list.

For TypeScript callers, use `compactMessages(messages, options)` from this
package. Supply a custom `JevAsker` when the host owns networking. Select the
TypeSafe transport only with an explicit `provider: 'typesafe'`; it is not the
default or fallback.
