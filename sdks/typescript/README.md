# @forgefy/sdk

Official TypeScript SDK for the [Forgefy Developer API](../../docs/developer-api.md) —
transcripts in, structured product requirements (features, open questions,
conflicts, action items) out.

Node 18+. Zero runtime dependencies.

```bash
npm install @forgefy/sdk
```

## Quick start

```ts
import Forgefy from "@forgefy/sdk";

const forgefy = new Forgefy({
  apiKey: process.env.FORGEFY_API_KEY!, // fgy_live_… from the dashboard's Developers page
  baseUrl: "https://your-forgefy-host", // or set $FORGEFY_API_URL
});

const result = await forgefy.extract({
  transcript: "We need Google login before launch. Sarah owns billing.",
  extractors: ["features", "action_items"], // optional — omit to run all four
});

for (const f of result.features) console.log(`[${f.priority}] ${f.title}`);
console.log(result.usage); // { input_tokens, output_tokens }
```

## Long transcripts (async jobs)

```ts
const job = await forgefy.jobs.create({
  transcript: longTranscript, // up to 200k chars
  webhookUrl: "https://yourapp.com/hooks/forgefy", // optional
});

// Either wait by polling…
const done = await forgefy.jobs.waitFor(job.job_id); // throws JobFailedError / JobTimeoutError
console.log(done.result);

// …or verify the webhook delivery instead:
import { verifySignature } from "@forgefy/sdk";

app.post("/hooks/forgefy", (req, res) => {
  // req.rawBody must be the raw bytes — verify BEFORE parsing JSON
  if (!verifySignature(req.rawBody, req.headers["x-forgefy-signature"], job.webhook_secret!)) {
    return res.status(401).end();
  }
  const event = JSON.parse(req.rawBody); // { type, job_id, status, result? , error? }
  res.status(200).end();
});
```

`jobs.create()` sends an auto-generated `Idempotency-Key`, so a network-level
retry can never run the same job twice. Pass `idempotencyKey` yourself to
dedupe across processes.

## Errors

All API failures throw typed subclasses of `ForgefyError` with `.status` and `.detail`:

| Class | When |
|---|---|
| `AuthenticationError` | 401 — bad or revoked key |
| `QuotaExceededError` | 402 — monthly tokens exhausted (free tier); `detail` has the reset date |
| `NotFoundError` | 404 |
| `ValidationError` | 422 |
| `RateLimitError` | 429 — 60 req/min per key (retried automatically first) |
| `ServerError` | 5xx |
| `APIConnectionError` | no HTTP response at all |

Retry behavior: 429 and network errors are always retried (default 2 retries,
exponential backoff, `Retry-After` honored). 5xx is retried only where safe —
GETs and idempotent job creation, never the sync `extract()` (a retry would
bill tokens twice).

## Quota

```ts
const usage = await forgefy.usage();
// { tier, monthly_tokens, tokens_used, tokens_remaining, resets_at }
```

Paid accounts over budget aren't blocked — requests are served by the free
`economy` model instead. Check `result.model_tier` to see which tier answered.

## Development

```bash
npm install
npm test        # vitest, all HTTP mocked
npm run build   # emits dist/
```
