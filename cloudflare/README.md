# Cloudflare Workers + Containers — optional deployment adapter

This directory is an **additional deployment target**, not a replacement for anything.
`docker-compose.yml` remains the source of truth for local dev and other platforms.

Nothing outside this directory is Cloudflare-aware:

- `../Dockerfile` and `../Dockerfile.worker` are used **as-is**, unmodified
- No app code imports a Cloudflare API
- Env vars keep their existing names (`REDIS_URL`, `CELERY_BROKER_URL`, …)
- `../docker-compose.yml` is untouched

**To drop Cloudflare as a target: delete this directory.** Nothing else changes.

| File | Purpose |
| --- | --- |
| `wrangler.json` | Container + Durable Object + migration config |
| `src/index.ts` | Worker entry: `ApiContainer`, `WorkerContainer`, routing, supervisor |
| `src/container-env.ts` | Env pass-through (deny-list, no renames) |
| `scripts/push-secrets.ps1` | Uploads `../.env` via `wrangler secret bulk` |

## Architecture

```
                       ┌──────────────────────────────┐
  HTTPS / WSS  ────────▶  Worker (src/index.ts)       │
                       │  getRandom(API_CONTAINER, 5) │
                       └───────┬──────────────────────┘
                               │ stub.fetch()  (WS-safe)
                  ┌────────────▼─────────────┐
                  │ ApiContainer ×≤5         │   ../Dockerfile
                  │ uvicorn :8000            │   standard-2
                  │ sleepAfter 1h            │
                  └────────────┬─────────────┘
                               │
     cron */2min               │  redis://…:11801
  ┌──────────────┐             ▼
  │ scheduled()  │      ┌─────────────┐        ┌──────────────────────┐
  │ ensureRunning├─────▶│ Redis Cloud │◀───────┤ WorkerContainer ×1   │
  └──────────────┘      └─────────────┘        │ celery -B            │
                                               │ ../Dockerfile.worker │
                                               │ standard-4, no sleep │
                                               └──────────────────────┘
```

The API and the worker never talk to each other directly — they communicate
through Redis Cloud exactly as they do under Docker Compose.

## How the worker stays alive

`sleepAfter` counts **inbound requests**. Nothing ever sends the Celery worker a
request; it polls Redis outbound. So its idle timer always expires.

Cloudflare's documented behaviour: if `onActivityExpired()` does **not** call
`stop()` or `destroy()`, the timer renews and the hook fires again at the next
expiry. `WorkerContainer.onActivityExpired()` in `src/index.ts` deliberately does
nothing but log and renew — so the container never sleeps.

That covers idle shutdown, but not everything. Cloudflare explicitly gives **no
guarantee that a container instance runs for any set period** — a host restart
can stop it. The cron trigger (every 2 minutes) calls `ensureRunning()`, which
restarts it if it is not running. Both mechanisms are needed.

## Deploy

Requires: Workers **Paid** plan, Docker running locally, Node 18+.

```powershell
cd cloudflare
npm install

npx wrangler login

# Upload ../.env as Worker secrets, names unchanged. Check first:
./scripts/push-secrets.ps1 -DryRun
./scripts/push-secrets.ps1

# Guards the /__cf/* admin routes. Without it they return 503.
npx wrangler secret put CF_ADMIN_TOKEN

npx wrangler deploy
```

That single `wrangler deploy` builds both Dockerfiles, pushes both images to the
Cloudflare Registry, and deploys the Worker.

> **The first deploy is slow.** Both images are ~8.23 GB. Common base layers are
> content-addressed and pushed once, but expect well over 10 GB of upload on a
> cold cache. Subsequent deploys only push changed layers.

Later deploys are just `npx wrangler deploy`. Secrets persist across deploys —
re-run `push-secrets.ps1` only when `.env` changes.

### What the dry run found in your `.env`

- **68 unique keys upload**, comfortably under wrangler's 100-per-request limit.
- **`OPENAI_API_KEY` is defined twice.** The script keeps the last occurrence
  (matching shell and `env_file` semantics) and warns. Worth cleaning up in
  `.env` regardless, since the two values may differ.
- Several keys are in `.env` but not `.env.example` — `CLOUDINARY_URL`,
  `FIREBASE_WEB_API_KEY`, `NEON_ORG_ID`, `OLLAMA_BUILD_MODEL`. They upload
  automatically: `src/container-env.ts` uses a deny-list, not an allow-list, so
  new settings need no change here. Consider adding them to `.env.example`.
- Only `APP_ENV` was skipped as already-declared-in-`vars`. `PORT`,
  `WEB_CONCURRENCY` and `PYTHONUNBUFFERED` are not in your `.env` at all, so
  the `vars` values apply cleanly.

## Verify after deploy

Set `$URL` to the `workers.dev` URL that `wrangler deploy` prints, and
`$TOKEN` to your `CF_ADMIN_TOKEN`.

**1. API responds**

```powershell
curl "$URL/health"
```

Expect the FastAPI liveness/readiness payload from `app/main.py:197`. First hit
is a cold start and may take ~30–60s while the container boots; after that it
should be fast.

**2. WebSockets upgrade** (this is what breaks if routing ever gets switched to
`containerFetch`)

```powershell
npx wscat -c "$URL/ws/projects" 
```

Expect a completed upgrade, not a 426/500.

**3. Worker container is running**

```powershell
curl -H "x-cf-admin-token: $TOKEN" "$URL/__cf/status"
```

Expect `{"worker":{"status":"running"}}`. If it reports `stopped`, start it
without waiting for the cron:

```powershell
curl -X POST -H "x-cf-admin-token: $TOKEN" "$URL/__cf/worker/start"
```

**4. Worker is actually consuming from Redis Cloud** — the important one, since
a container can be "running" while Celery fails to reach the broker.

```powershell
npx wrangler tail --format pretty
```

Look for Celery's banner listing the broker and the four queues
(`meeting.audio`, `meeting.transcribe`, `meeting.extract`, `build`), then
`celery@… ready.`. A repeating `consumer: Cannot connect to redis://…` means
outbound TCP to port 11801 is not working — see the checklist item below.

Confirm from the Redis side too:

```
redis-cli -u "$REDIS_URL" CLIENT LIST
```

> The current Redis Cloud database has **TLS disabled** — the endpoint speaks
> plain RESP on 11801 and `rediss://` fails with `WRONG_VERSION_NUMBER`. If you
> enable TLS on the database, switch the three `.env` URLs back to `rediss://`
> and re-run `push-secrets.ps1`; `app/config.py` and `celery_app.py` already
> handle the `rediss://` case (they append `ssl_cert_reqs=none` and set
> `broker_use_ssl`), so no code change is needed.

You should see connections from a Cloudflare egress IP. Then enqueue a real job
through the API and watch it get picked up in `wrangler tail`.

**5. It does not sleep after 10 minutes** — the specific thing you asked about.

Leave it idle for **>15 minutes** with no requests, then:

```powershell
curl -H "x-cf-admin-token: $TOKEN" "$URL/__cf/status"
```

Still `running` = the `onActivityExpired` override is working. In `wrangler tail`
you should see `[WorkerContainer] activity window expired — keeping alive for
Celery` roughly every 12 hours, and **no** `[WorkerContainer] stopped` line.

If it comes back `stopped`, the override is not taking effect — check that
`onActivityExpired` has not been removed and does not call `stop()`.

**6. Image storage headroom**

```powershell
npx wrangler containers images list
```

Account cap is 50 GB across all images and retained versions. Two ~8.23 GB
images plus history will consume it faster than you expect.

## Findings on your existing Dockerfiles

**I did not change either file.** These are things to be aware of, in rough
order of how likely they are to bite.

### 1. Image size drives instance type — ~8.23 GB each

Cloudflare caps image size at the **instance's disk size**. Flutter 3.24.4 +
Android SDK 34 + JDK 21 + Node 22 + Playwright Chromium put both images at
8.23 GB, which rules out `lite` (2 GB), `basic` (4 GB) and `standard-1` (8 GB).

| | instance | disk | scratch left after image |
| --- | --- | --- | --- |
| API | `standard-2` | 12 GB | ~3.8 GB |
| Worker | `standard-4` | 20 GB | ~11.8 GB |

The worker needs the larger figure because Gradle/Flutter builds write into
`/tmp/forgefy_workspaces`. If builds start failing on disk, `standard-4` is
already the ceiling (20 GB) — the image has to get smaller at that point.

### 2. Disk is ephemeral and wiped on sleep or restart

`WORKSPACE_ROOT = Path("/tmp/forgefy_workspaces")` (`app/build/workspace.py:15`).
Everything under it vanishes when a container stops. Two consequences:

- **A build interrupted by a restart loses its workspace.** Anything that must
  survive has to be in Redis, Cloudinary, or object storage before the task ends.
- **`/tmp` is per-instance.** With 5 API instances, a workspace written by one is
  invisible to the other four. This is already true under Compose (`/tmp` is not
  a shared volume), so it should not be new behaviour — but scaling the API from
  1 to 5 makes it much more likely to surface.

Also `celery -B -s /tmp/celerybeat-schedule` (`../Dockerfile.worker:65`): the beat
schedule DB is on ephemeral disk, so after a restart beat has no memory of when
it last ran and may re-fire schedules. Under Compose this file persists in the
repo root — you have `celerybeat-schedule`, `-shm` and `-wal` files there now.

### 3. No `.dockerignore` — and a credential in the build context

There is no `.dockerignore`, so `COPY . .` pulls the entire directory into both
images, including:

`venv/` · `.git/` · `.mypy_cache/` · `.pytest_cache/` · `.ruff_cache/` ·
`.coverage` · `celerybeat-schedule*` · **`firebase-credentials.json`**

That last one bakes a live credential into an image layer. Adding a
`.dockerignore` would shrink both images meaningfully and speed up every deploy —
**and it would speed up your Compose builds too**. I did not add one because it
changes what your other platforms build. Say the word and I will.

### 4. `HEALTHCHECK` is ignored

`../Dockerfile:72` defines a `HEALTHCHECK`. Cloudflare Containers do not run it —
readiness is determined by TCP port checks (`startAndWaitForPorts`). Harmless,
just inert. Compose still uses it.

### 5. Port: 8000, not 5000

`../Dockerfile` has `EXPOSE 8000` and defaults to `${PORT:-8000}`, but
`docker-compose.yml` overrides the command to `--port 5000`. The adapter uses
**8000**, matching the image's own default, and sets `PORT=8000` in `vars`.

### 6. `WEB_CONCURRENCY` lowered to 2

The Dockerfile defaults to 4 uvicorn workers. On a 6 GiB `standard-2`, four
processes each importing playwright/firebase/the AI SDKs is heavy on memory and
slows cold start. `vars` sets `WEB_CONCURRENCY=2`. Raise it in `wrangler.json` if
you move the API to a bigger instance — no Dockerfile change needed, which is why
that CMD uses shell form.

### 7. Cold start can exceed the default 20s port-ready window

Given the image size and import graph, `ApiContainer.fetch()` explicitly boots
with `portReadyTimeoutMS: 60_000` instead of the 20s default. If you still see
start timeouts, lower `WEB_CONCURRENCY` further or raise that number.

### 8. `linux/amd64` required

Cloudflare runs `linux/amd64` only. You are building on Windows/x86_64, so this
is automatic. It would only matter if you moved builds to an ARM machine, where
you would need `--platform linux/amd64`.

### 9. Docker socket path would not work — but is dormant

`app/connectors/zoom_selfhosted.py:158` calls `docker.from_env()`, and
`docker-compose.yml` mounts `/var/run/docker.sock` into the worker. **There is no
host Docker socket on Cloudflare Containers**, so that code path cannot work
there.

Your `.env` has `ZOOM_BOT_PROVIDER=recall`, so it is not exercised. **Do not set
`ZOOM_BOT_PROVIDER=self_hosted` on the Cloudflare deployment** — it will fail at
runtime. (Cloudflare supports `docker:dind-rootless` with iptables disabled, but
that is a different and much heavier design than socket mounting.)

### 10. One worker replica, because beat is embedded

`../Dockerfile.worker:65` runs `celery … -B`, i.e. beat inside the worker. That is
why `max_instances` is **1** — a second replica would double-fire every scheduled
task. Under Compose you avoid this by running a separate `beat` service and a
worker without `-B`.

To scale worker throughput: raise `CELERY_CONCURRENCY` (currently 2) rather than
`max_instances`. Going past one replica means splitting beat into its own
container with an `entrypoint` override — ask and I will add it.

## Cost

Ballpark for the always-on worker, at published rates:

| | monthly |
| --- | --- |
| Worker memory (12 GiB, always on) | ~$79 |
| Worker disk (20 GB, always on) | ~$4 |
| Worker CPU | usage-based; idle Celery is cheap, Flutter builds are not |
| API | scales to zero after 1h idle; billed per running instance |

The `standard-4` always-on floor is roughly **$83/month before CPU**, because
memory and disk bill on *provisioned* resources for as long as the instance is
running. CPU bills on actual use. Sustained 4-vCPU builds could add
~$200/month on their own, so watch the first bill.

This is the direct cost of the "never sleeps" requirement. If the queue is idle
most of the day, an alternative is to let the worker sleep and start it on demand
from a Cron Trigger or an API call — say the word and I will wire that instead.

## Known limitations

- **WebSockets and sleep.** `ApiContainer.sleepAfter = "1h"`. A socket held open
  with no new HTTP requests to that instance for an hour can be dropped when the
  container sleeps. Clients should reconnect; state is rebuilt from Redis pub/sub.
  For guaranteed-open sockets, give `ApiContainer` the same `onActivityExpired`
  override as the worker — at always-on cost per instance.
- **Containers are in beta.** No SLA, API may change, rolling deploys only.
- **`wrangler.json` lives here, not at the repo root.** You asked for both; the
  isolation constraint won. Every command needs `-c cloudflare/wrangler.json` if
  run from the repo root, or just `cd cloudflare` first. Paths inside the config
  (`../Dockerfile`, `image_build_context: ".."`) are relative to the config file.
- **`CLOUDFLARE_API_TOKEN` collision.** Your `.env` has one for the app's own use.
  Wrangler authenticates with whatever is exported in your *shell*, so a stale
  value there will target the wrong account. `push-secrets.ps1` warns about this.
