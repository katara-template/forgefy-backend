# forgefy

Official Python SDK for the [Forgefy Developer API](../../docs/developer-api.md) —
transcripts in, structured product requirements (features, open questions,
conflicts, action items) out.

Python 3.10+. One dependency (`httpx`).

```bash
pip install forgefy
```

## Quick start

```python
import os
from forgefy import Forgefy

client = Forgefy(
    api_key=os.environ["FORGEFY_API_KEY"],  # fgy_live_… from the dashboard's Developers page
    base_url="https://your-forgefy-host",   # or set $FORGEFY_API_URL
)

result = client.extract(
    "We need Google login before launch. Sarah owns billing.",
    extractors=["features", "action_items"],  # optional — omit to run all four
)

for f in result["features"]:
    print(f"[{f['priority']}] {f['title']}")
print(result["usage"])  # {"input_tokens": ..., "output_tokens": ...}
```

## Long transcripts (async jobs)

```python
job = client.jobs.create(
    long_transcript,  # up to 200k chars
    webhook_url="https://yourapp.com/hooks/forgefy",  # optional
)

# Either wait by polling…
done = client.jobs.wait_for(job["job_id"])  # raises JobFailedError / JobTimeoutError
print(done["result"])

# …or verify the webhook delivery instead:
from forgefy import verify_signature

@app.post("/hooks/forgefy")
async def hook(request):
    raw = await request.body()  # raw bytes — verify BEFORE parsing JSON
    if not verify_signature(raw, request.headers.get("x-forgefy-signature"), job["webhook_secret"]):
        return Response(status_code=401)
    event = json.loads(raw)  # {"type", "job_id", "status", "result"?, "error"?}
    return Response(status_code=200)
```

`jobs.create()` sends an auto-generated `Idempotency-Key`, so a network-level
retry can never run the same job twice. Pass `idempotency_key=` yourself to
dedupe across processes.

## Errors

All API failures raise typed subclasses of `ForgefyError` with `.status` and `.detail`:

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

```python
usage = client.usage()
# {"tier", "monthly_tokens", "tokens_used", "tokens_remaining", "resets_at"}
```

Paid accounts over budget aren't blocked — requests are served by the free
`economy` model instead. Check `result["model_tier"]` to see which tier answered.

## Development

```bash
# from forgefy-backend/, using its venv (httpx + pytest already installed)
python -m pytest sdks/python/tests -q
```
