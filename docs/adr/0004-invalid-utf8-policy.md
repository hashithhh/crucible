# ADR-0004 — Decoding invalid UTF-8

- **Status:** accepted
- **Date:** 2026-09-17

## Context
Token ids map to byte strings, and any list of valid ids can be concatenated.
Not every concatenation is valid UTF-8 — a caller can hand `decode` the ids
for a lone continuation byte, or slice a token sequence mid-character.

## Options considered
| Option | Behaviour | Note |
|---|---|---|
| `errors="strict"` | raises UnicodeDecodeError | makes decode partial; callers must guard every call |
| `errors="replace"` | emits U+FFFD | total, lossy, visible in output |
| `errors="surrogateescape"` | round-trips the bytes | total and lossless, but produces strings that cannot be re-encoded as UTF-8 |

## Decision
**`errors="replace"`.**

## Rationale
`decode` must stay total: it is called on model output, which during early
training is effectively random ids and will routinely produce invalid
sequences. A tokenizer that raises there turns a normal training artifact
into a crash. `surrogateescape` is lossless but the strings it produces fail
on re-encode, which moves the failure somewhere harder to find.

Lossy is acceptable because the lossless path that matters —
`decode(encode(s)) == s` for real text — is unaffected and is covered by the
round-trip tests.

## Consequences
Pinned by `test_decode_invalid_utf8_policy`. Out-of-range ids are a different
case and still raise `ValueError`.
