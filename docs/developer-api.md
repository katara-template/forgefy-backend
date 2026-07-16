# Forgefy Developer API

Turn meeting transcripts into structured product requirements — features, open
questions, conflicting requirements, and action items — over a REST API.

Base URL: `https://<your-forgefy-host>/api/v1`
Interactive OpenAPI docs: `/docs` (development builds).

## Authentication

1. Sign in to the Forgefy dashboard and create an API key (`POST /keys`, or the
   dashboard UI). The full key (`fgy_live_…`) is shown **once** — store it
   securely. Only a SHA-256 hash is kept server-side.
2. Send it on every request:

```
Authorization: Bearer fgy_live_...
```

Keys can be listed (prefixes only) and revoked at any time; revocation takes
effect within ~30 seconds.

## Quickstart — synchronous extraction

Transcripts up to 50,000 characters return in one request. The four
extractors run in parallel, so latency is roughly a single model call.

```bash
curl -X POST "$FORGEFY_URL/api/v1/extract" \
  -H "Authorization: Bearer $FORGEFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "We need Google login before launch. Sarah will own the billing page. Mobile is out of scope for v1 — wait, Tom said mobile-first last week.",
    "extractors": ["features", "action_items", "conflicts"]
  }'
```

```json
{
  "id": "6d5e…",
  "model_tier": "standard",
  "features": [
    {"title": "Google OAuth login", "description": "…", "priority": "high"}
  ],
  "questions": [],
  "conflicts": [
    {"description": "Mobile scope disagreement", "side_a": "…", "side_b": "…"}
  ],
  "action_items": [
    {"task": "Build the billing page", "owner": "Sarah", "due": null}
  ],
  "usage": {"input_tokens": 412, "output_tokens": 188}
}
```

Request fields:

| Field | Type | Notes |
|---|---|---|
| `transcript` | string, required | ≤ 50k chars (sync) / ≤ 200k chars (jobs) |
| `extractors` | string[] | Any of `features`, `questions`, `conflicts`, `action_items`. Omit for all four. You are only metered for the extractors you request. |
| `model_tier` | string | `standard` (Claude, metered) or `economy` (Qwen3, free but slower/lower fidelity) |

### Python

```python
import httpx

resp = httpx.post(
    f"{FORGEFY_URL}/api/v1/extract",
    headers={"Authorization": f"Bearer {FORGEFY_API_KEY}"},
    json={"transcript": transcript, "extractors": ["features", "questions"]},
    timeout=120,
)
resp.raise_for_status()
for feature in resp.json()["features"]:
    print(f'[{feature["priority"]}] {feature["title"]}')
```

### JavaScript

```js
// Node 18+ or the browser — fetch is built in
const resp = await fetch(`${FORGEFY_URL}/api/v1/extract`, {
  method: "POST",
  headers: {
    Authorization: `Bearer ${process.env.FORGEFY_API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ transcript, extractors: ["action_items"] }),
});
if (!resp.ok) throw new Error(`Forgefy error ${resp.status}`);
const { action_items } = await resp.json();
```

### Go

```go
payload, _ := json.Marshal(map[string]any{
	"transcript": transcript,
	"extractors": []string{"features", "action_items"},
})

req, _ := http.NewRequest("POST", forgefyURL+"/api/v1/extract", bytes.NewReader(payload))
req.Header.Set("Authorization", "Bearer "+os.Getenv("FORGEFY_API_KEY"))
req.Header.Set("Content-Type", "application/json")

resp, err := http.DefaultClient.Do(req)
if err != nil {
	log.Fatal(err)
}
defer resp.Body.Close()

var out struct {
	Features []struct {
		Title    string `json:"title"`
		Priority string `json:"priority"`
	} `json:"features"`
}
json.NewDecoder(resp.Body).Decode(&out)
```

### PHP

```php
<?php
$ch = curl_init($forgefyUrl . "/api/v1/extract");
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST => true,
    CURLOPT_HTTPHEADER => [
        "Authorization: Bearer " . getenv("FORGEFY_API_KEY"),
        "Content-Type: application/json",
    ],
    CURLOPT_POSTFIELDS => json_encode([
        "transcript" => $transcript,
        "extractors" => ["features", "action_items"],
    ]),
]);
$data = json_decode(curl_exec($ch), true);
curl_close($ch);
```

### Mobile (Swift / Kotlin / Dart)

> **Shipping to an app store?** Anyone can extract strings from your binary —
> keep the `fgy_live_` key on your own server and have the app call that
> instead. Calling Forgefy directly from the device is fine for prototypes and
> internal tools.

**Swift** (iOS/macOS, `URLSession`, no dependencies):

```swift
struct ExtractResponse: Decodable {
    struct Feature: Decodable { let title: String; let priority: String }
    let features: [Feature]
}

var request = URLRequest(url: URL(string: "\(forgefyURL)/api/v1/extract")!)
request.httpMethod = "POST"
request.setValue("Bearer fgy_live_...", forHTTPHeaderField: "Authorization")
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.httpBody = try JSONSerialization.data(withJSONObject: [
    "transcript": transcript,
    "extractors": ["features", "action_items"],
])

let (data, response) = try await URLSession.shared.data(for: request)
guard (response as? HTTPURLResponse)?.statusCode == 200 else {
    throw URLError(.badServerResponse)
}
let result = try JSONDecoder().decode(ExtractResponse.self, from: data)
```

**Kotlin** (Android, OkHttp):

```kotlin
val payload = JSONObject().apply {
    put("transcript", transcript)
    put("extractors", JSONArray(listOf("features", "action_items")))
}

val request = Request.Builder()
    .url("$forgefyUrl/api/v1/extract")
    .header("Authorization", "Bearer ${BuildConfig.FORGEFY_API_KEY}")
    .post(payload.toString().toRequestBody("application/json".toMediaType()))
    .build()

OkHttpClient().newCall(request).execute().use { response ->
    check(response.isSuccessful) { "Forgefy error ${response.code}" }
    val data = JSONObject(response.body!!.string())
}
```

**Dart / Flutter** (`http` package):

```dart
final resp = await http.post(
  Uri.parse('$forgefyUrl/api/v1/extract'),
  headers: {
    'Authorization': 'Bearer fgy_live_...',
    'Content-Type': 'application/json',
  },
  body: jsonEncode({
    'transcript': transcript,
    'extractors': ['features', 'action_items'],
  }),
);
if (resp.statusCode != 200) {
  throw Exception('Forgefy error ${resp.statusCode}');
}
final data = jsonDecode(resp.body) as Map<String, dynamic>;
```

## Async jobs — long transcripts

For transcripts up to 200k characters (a full workday of speech), queue a job
and either poll or receive a webhook.

```bash
curl -X POST "$FORGEFY_URL/api/v1/extract/jobs" \
  -H "Authorization: Bearer $FORGEFY_API_KEY" \
  -H "Idempotency-Key: import-batch-42" \
  -d '{
    "transcript": "…",
    "webhook_url": "https://yourapp.com/hooks/forgefy"
  }'
# → 202 {"job_id": "…", "status": "queued", "webhook_secret": "…"}

curl "$FORGEFY_URL/api/v1/extract/jobs/$JOB_ID" \
  -H "Authorization: Bearer $FORGEFY_API_KEY"
# → {"status": "done", "result": {"features": […], "usage": {…}}, …}
```

- **Idempotency**: replaying a request with the same `Idempotency-Key` returns
  the existing job instead of running a second one. Use it for safe retries.
- **Retention**: job records are deleted 30 days after creation.

### Webhooks

When the job finishes, its result is POSTed to `webhook_url` (https required):

```json
{"type": "extract.job.completed", "job_id": "…", "status": "done", "result": {…}}
{"type": "extract.job.failed",    "job_id": "…", "status": "failed", "error": "…"}
```

Deliveries are signed with the `webhook_secret` returned at job creation:

```
X-Forgefy-Signature: sha256=<HMAC-SHA256 hex of the raw request body>
```

Verify before trusting the payload:

```python
import hashlib, hmac

def verify(raw_body: bytes, signature_header: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature_header, f"sha256={expected}")
```

Failed deliveries are retried 3 times with exponential backoff. Delivery is
independent of extraction — a flaky receiver never re-runs the pipeline.

## Usage & quotas

API usage draws from the same monthly token budget as the rest of your
Forgefy plan (see the tier table in the dashboard). Check consumption
programmatically:

```bash
curl "$FORGEFY_URL/api/v1/usage" -H "Authorization: Bearer $FORGEFY_API_KEY"
```

```json
{
  "tier": "starter",
  "monthly_tokens": 5000000,
  "tokens_used": 1200000,
  "tokens_remaining": 3800000,
  "resets_at": "2026-08-01T00:00:00+00:00"
}
```

Over-budget behavior:

- **Free tier** → requests fail with `402` and a reset date.
- **Paid tiers** → requests are transparently served by the `economy` tier
  (unmetered) instead of failing — the response's `model_tier` tells you which
  tier actually served it.

## Errors

Errors are RFC 7807 `application/problem+json`:

| Status | Meaning |
|---|---|
| `401` | Missing, invalid, or revoked API key |
| `402` | Monthly token budget exhausted (free tier) |
| `404` | Job doesn't exist or belongs to another account |
| `422` | Validation error (empty/oversized transcript, bad extractor name, non-https webhook) |
| `429` | Rate limited (60 requests/minute **per API key**), or the server is at sync-extraction capacity — back off and retry, or switch to `/extract/jobs` |
| `502` | Extraction failed on every model, or the job queue is unreachable |

## MCP server

`scripts/mcp_server.py` is a small local connector that exposes the API as MCP
tools, so agents (Claude Code, Claude Desktop, Cursor, …) can call Forgefy
directly — no HTTP code on your side. It is a single self-contained file; end
users can also download it from the web app at `https://<frontend-host>/mcp_server.py`.

**1. Get the file and its dependency** (Python 3.10+):

```bash
pip install "mcp[cli]" httpx
```

**2. Register it with your client.** You do not run the server yourself — the
MCP client spawns it (over stdio) whenever a session needs it.

Claude Code (terminal):

```bash
claude mcp add forgefy \
  -e FORGEFY_API_KEY=fgy_live_... \
  -e FORGEFY_API_URL=https://<your-forgefy-host> \
  -- python /path/to/mcp_server.py
```

Claude Desktop / Cursor (`claude_desktop_config.json` / `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "forgefy": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "FORGEFY_API_KEY": "fgy_live_...",
        "FORGEFY_API_URL": "https://<your-forgefy-host>"
      }
    }
  }
}
```

**3. Use it.** Restart the client (in Claude Code, `/mcp` should list
`forgefy` as connected), then ask in plain language — *"Use forgefy to extract
the requirements from this transcript: …"*. The agent picks the tool itself.

Tools: `extract_requirements`, `create_extract_job`, `get_extract_job`,
`get_usage`.

To debug the server interactively without an editor:

```bash
npx @modelcontextprotocol/inspector python scripts/mcp_server.py
```
