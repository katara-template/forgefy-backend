# forgefy

Official Dart & Flutter SDK for the [Forgefy Developer API](../../docs/developer-api.md) —
transcripts in, structured product requirements (features, open questions,
conflicts, action items) out.

Pure Dart — works on the Dart VM, Flutter mobile/desktop, and Flutter web
(depends only on `package:http`, never `dart:io`).

```yaml
dependencies:
  forgefy: ^0.1.0
```

## Quick start

```dart
import 'package:forgefy/forgefy.dart';

final forgefy = Forgefy(
  apiKey: 'fgy_live_...',            // from the dashboard's Developers page
  baseUrl: 'https://your-forgefy-host',
);

final result = await forgefy.extract(
  transcript: 'We need Google login before launch. Sarah owns billing.',
  extractors: [Extractor.features, Extractor.actionItems], // optional — omit for all four
);

for (final f in result.features) {
  print('[${f.priority}] ${f.title}');
}
print(result.usage.inputTokens); // token consumption

forgefy.close(); // release the HTTP client when done
```

Every model-generated payload keeps its full JSON on `.raw`, so fields not
typed on `Feature` / `Question` / `Conflict` / `ActionItem` are still reachable.

## Long transcripts (async jobs)

```dart
final job = await forgefy.jobs.create(
  transcript: longTranscript,                      // up to 200k chars
  webhookUrl: 'https://yourapp.com/hooks/forgefy', // optional
);

// Either wait by polling…
final done = await forgefy.jobs.waitFor(job.jobId); // throws JobFailedException / JobTimeoutException
print(done.result);

// …or verify the webhook delivery instead (see below).
```

`jobs.create()` sends an auto-generated `Idempotency-Key`, so a network-level
retry can never run the same job twice. Pass `idempotencyKey:` yourself to
dedupe across processes.

## Webhooks

Deliveries are signed `X-Forgefy-Signature: sha256=<hex>`, HMAC-SHA256 over the
raw body with the `webhookSecret` returned at job creation. Verify against the
**raw** bytes, before parsing JSON:

```dart
import 'package:forgefy/forgefy.dart';

// rawBody: the exact bytes received; signature: the X-Forgefy-Signature header
if (!verifySignature(rawBody, signature, job.webhookSecret!)) {
  // reject with 401
}
```

Comparison is constant-time; malformed input returns `false` rather than throwing.

## Errors

Every API failure throws a subclass of `ForgefyException` with `.status` and `.detail`:

| Class | When |
|---|---|
| `AuthenticationException` | 401 — bad or revoked key |
| `QuotaExceededException` | 402 — monthly tokens exhausted (free tier); `detail` has the reset date |
| `NotFoundException` | 404 |
| `ValidationException` | 422 |
| `RateLimitException` | 429 — 60 req/min per key (retried automatically first) |
| `ServerException` | 5xx |
| `ApiConnectionException` | no HTTP response at all |
| `JobFailedException` / `JobTimeoutException` | `jobs.waitFor` outcomes |

Retry behavior mirrors the other SDKs: 429 and network errors are always
retried (default 2 retries, exponential backoff with jitter, `Retry-After`
honored). 5xx is retried only where safe — GETs and idempotent job creation,
never the sync `extract()` (a retry would bill tokens twice).

## Quota

```dart
final usage = await forgefy.usage();
// usage.tier, usage.monthlyTokens, usage.tokensUsed, usage.tokensRemaining, usage.resetsAt
```

Paid accounts over budget aren't blocked — requests are served by the free
`economy` model instead. Check `result.modelTier` to see which tier answered.

## Development

```bash
dart pub get
dart test        # all HTTP mocked
dart analyze
```
