# Changelog

## 0.1.0

Initial release. Mirrors the TypeScript and Python SDKs.

- `Forgefy.extract()` — synchronous extraction (transcripts ≤ 50k chars).
- `Forgefy.jobs.create()` / `.get()` / `.waitFor()` — async extraction jobs
  (transcripts up to 200k chars) with an auto-generated `Idempotency-Key`.
- `Forgefy.usage()` — the key owner's tier and monthly token budget.
- Typed exception hierarchy mapped from the API's problem+json responses.
- `verifySignature()` — constant-time verification of signed webhook deliveries.
- Automatic retries for 429 and transient network failures (exponential
  backoff with jitter, `Retry-After` honored).
