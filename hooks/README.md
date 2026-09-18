# Legacy Claude function-hook adapter

`fast-jev.ts` adapts a host that exposes `session.compact` and
`turn.complete` function hooks. It now defaults to OpenRouter with
`OPENROUTER_API_KEY` and `~typesafe/jev-latest`; select `provider: "typesafe"`
only for an explicit private TypeSafe key.

The adapter fails open when a request times out, a response is invalid, the
classifier state is incomplete, or reduction is too small. It also acquires its
auto-compaction lock before asynchronous work and treats logging and toasts as
best-effort diagnostics.

This directory is retained for experimental hosts that implement those function
hook events. Public Claude Code installations should not treat it as a supported
plugin: their released hook surface does not replace the session transcript.
