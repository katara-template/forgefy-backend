/**
 * Cloudflare Workers + Containers deployment adapter for forgefy-backend.
 *
 * This file is the ONLY Cloudflare-specific runtime code. The app itself has no
 * idea it is running here: it sees the same env var names, the same ports and
 * the same commands as under Docker Compose. Delete the ./cloudflare directory
 * and every other deployment target is unaffected.
 *
 *   ApiContainer    — ./Dockerfile,        uvicorn on :8000, request-routed
 *   WorkerContainer — ./Dockerfile.worker, celery -B, always running
 */

import { Container, getContainer, getRandom } from "@cloudflare/containers";

import {
  API_POOL_SIZE,
  API_PORT,
  WORKER_INSTANCE_NAME,
  assertRequiredEnv,
  buildContainerEnv,
} from "./container-env";

export interface Env {
  API_CONTAINER: DurableObjectNamespace<ApiContainer>;
  WORKER_CONTAINER: DurableObjectNamespace<WorkerContainer>;
  /** Guards the /__cf/* admin routes. Set with `wrangler secret put`. */
  CF_ADMIN_TOKEN?: string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// API container
// ---------------------------------------------------------------------------

/**
 * Cold-start budget for the API container.
 *
 * `containerFetch()` calls `startAndWaitForPorts(port, { abort })` internally
 * with NO cancellationOptions, so it falls back to the library default of
 * 20s (TIMEOUT_TO_GET_PORTS_MS, @cloudflare/containers 0.3.7). uvicorn here
 * imports firebase / the AI SDKs at module scope in each of WEB_CONCURRENCY
 * workers and does not bind :8000 within that window, so every cold start
 * died in waitForPort before the app was ever listening.
 *
 * The fix is to run the wait ourselves with a realistic budget: once the
 * instance reaches `healthy`, containerFetch() skips its own wait entirely
 * and goes straight to proxying.
 *
 * Kept under Cloudflare's ~100s edge response limit on purpose — a larger
 * budget would just hand the client a 524 while this DO kept waiting. Boot
 * continues in the background after we give up, so the retry that the client
 * makes normally lands on a warm instance.
 */
const PORT_READY_TIMEOUT_MS = 85_000;

/** Time allowed to acquire and start the instance itself (library default: 8s). */
const INSTANCE_GET_TIMEOUT_MS = 30_000;

export class ApiContainer extends Container<Env> {
  defaultPort = API_PORT;

  /**
   * Built per Durable Object instance from `this.env`, not once per isolate.
   *
   * Field initializers run after `super(ctx, env)` has set `this.env`, and the
   * library reads `this.envVars` at container-start time (not in its
   * constructor), so this is safe. Every string-valued binding is passed
   * through under its own name — see ./container-env.ts.
   */
  private readonly containerEnv = buildContainerEnv(
    this.env as unknown as Record<string, unknown>,
  );

  /**
   * sleepMode: "afterIdle" equivalent. The Container class exposes this as
   * `sleepAfter` — the idle duration before the instance sleeps. After this
   * period with no requests the container is stopped (scales to zero) and the
   * next request triggers a cold start.
   *
   * Note the trade-off for WebSockets: a socket held open with no new HTTP
   * requests for this long can be dropped when the instance sleeps. Clients
   * should reconnect; the WS handlers rebuild state from Redis pub/sub anyway
   * (app/api/ws/connection_manager.py).
   *
   * Chosen to keep the container warm during active bursts (cold start here
   * costs ~90-150s) while still sleeping during genuine idle. Tune up/down
   * based on traffic patterns and cold-start tolerance.
   */
  sleepAfter = "10m";

  envVars = this.containerEnv;

  override onStart(): void {
    console.log(`[ApiContainer] uvicorn listening on :${API_PORT}`);
  }

  override onStop({ exitCode, reason }: { exitCode: number; reason: string }): void {
    console.log(`[ApiContainer] stopped exitCode=${exitCode} reason=${reason}`);
  }

  /**
   * Non-blocking cold start with retry.
   *
   * The previous implementation wrapped the heavy container start in
   * `this.ctx.blockConcurrencyWhile()`, which has a hard 30s limit. This
   * image's cold start (playwright / firebase / AI SDKs import at module
   * scope) routinely exceeds that, so the call was cancelled and the DO was
   * reset — surfacing as "container is not running".
   *
   * Fix: delegate to `containerFetch()`, the library's supported proxy path.
   * Internally it checks `this.container.running`, starts the container if
   * needed, and waits for the port — all OUTSIDE `blockConcurrencyWhile`
   * (the library only wraps the lightweight setHealthy()+onStart() in it).
   * This also preserves WebSocket upgrades on /ws/* (containerFetch accepts
   * both WebSocketPair ends), which the low-level port.fetch() path does not.
   *
   * Cold start is handled explicitly before the proxy: containerFetch()'s own
   * port wait is fixed at 20s, which this image cannot meet, so the wait runs
   * here with PORT_READY_TIMEOUT_MS instead. By the time we proxy, the port is
   * known ready — so the request is sent exactly once and whatever the app
   * answers is passed through untouched.
   */
  override async fetch(request: Request): Promise<Response> {
    const state = await this.getState();
    // Capture this up front: a 5xx is only retried when the container was
    // starting. If it was already running, a 5xx is an app-level error and
    // must be returned to the client as-is (never retried).
    const wasStarting = state.status !== "healthy" && state.status !== "running";
    console.log(`[ApiContainer] running status=${state.status}`);

    if (wasStarting) {
      // Check config before boot rather than after. uvicorn exits immediately
      // when SECRET_KEY is missing (app/main.py:96 refuses to serve production
      // traffic with the dev defaults), and a process that exits during boot
      // is indistinguishable from a slow one from waitForPort's side — you get
      // a port timeout 85s later with nothing pointing at the real cause.
      try {
        assertRequiredEnv(this.containerEnv);
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        console.error(`[ApiContainer] refusing to start: ${message}`);
        return new Response(JSON.stringify({ error: message }), {
          status: 503,
          headers: { "content-type": "application/json" },
        });
      }

      try {
        await this.startAndWaitForPorts({
          ports: API_PORT,
          cancellationOptions: {
            abort: request.signal,
            instanceGetTimeoutMS: INSTANCE_GET_TIMEOUT_MS,
            portReadyTimeoutMS: PORT_READY_TIMEOUT_MS,
          },
        });
        console.log(`[ApiContainer] :${API_PORT} ready, proxying`);
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        console.error(
          `[ApiContainer] :${API_PORT} not ready within ${PORT_READY_TIMEOUT_MS}ms: ${message}`,
        );
        // Boot carries on in the background; the client's retry usually lands
        // on a warm instance. Signal that rather than burning the retry loop
        // below on a container that is still importing.
        return new Response(
          JSON.stringify({
            error: "API container still starting — retry shortly",
            detail: message,
          }),
          { status: 503, headers: { "content-type": "application/json", "retry-after": "30" } },
        );
      }
    }

    // Single pass, deliberately. The old loop re-sent the request up to 5× on
    // any 5xx seen while the container was starting — which meant a POST could
    // be delivered to the app more than once (create-project, start-build and
    // the Stripe webhook are all non-idempotent). Now that the port is known
    // ready before we get here, a 5xx is the app's own answer and belongs to
    // the client unaltered.
    try {
      const response = await this.containerFetch(request, API_PORT);
      if (response.status >= 500) {
        console.log(`[ApiContainer] app returned status=${response.status}`);
      }
      return response;
    } catch (e) {
      // containerFetch returns a 5xx Response rather than throwing for the
      // startup/proxy cases, so a throw here is genuinely unexpected.
      const err = e instanceof Error ? e.message : String(e);
      console.error(`[ApiContainer] containerFetch threw: ${err}`);
      return new Response(
        JSON.stringify({ error: "API container proxy failed", detail: err }),
        { status: 502, headers: { "content-type": "application/json" } },
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Celery worker container
// ---------------------------------------------------------------------------

export class WorkerContainer extends Container<Env> {
  /**
   * No defaultPort: `celery worker` does not listen on anything. Starting this
   * container must therefore use start(), never startAndWaitForPorts().
   */
  sleepAfter = "12h";

  /** Per-instance, same as ApiContainer.containerEnv. */
  private readonly containerEnv = buildContainerEnv(
    this.env as unknown as Record<string, unknown>,
  );

  envVars = this.containerEnv;

  override onStart(): void {
    console.log("[WorkerContainer] celery worker started, consuming from Redis");
  }

  override onStop({ exitCode, reason }: { exitCode: number; reason: string }): void {
    // Not fatal: the cron supervisor restarts it within two minutes.
    console.log(`[WorkerContainer] stopped exitCode=${exitCode} reason=${reason}`);
  }

  /**
   * This is what keeps the worker alive.
   *
   * sleepAfter counts *inbound requests*, and nothing ever sends this container
   * a request — it polls Redis outbound. So the timer always expires. Per
   * Cloudflare's docs, an onActivityExpired() that does NOT call stop() or
   * destroy() causes the timer to renew and the hook to fire again at the next
   * expiry, so the container simply never sleeps.
   *
   * Deleting this override, or adding a stop() call to it, reintroduces the
   * 10-minute idle shutdown.
   */
  override async onActivityExpired(): Promise<void> {
    this.renewActivityTimeout();
    console.log("[WorkerContainer] activity window expired — keeping alive for Celery");
  }

  /**
   * Idempotent start, called every couple of minutes by the cron trigger.
   * Cloudflare can stop an instance for platform reasons (a host restart, for
   * one) and gives no runtime guarantee, so something has to notice and
   * restart it.
   */
  async ensureRunning(): Promise<{ status: string; restarted: boolean }> {
    this.renewActivityTimeout();

    const state = await this.getState();
    if (state.status === "running" || state.status === "healthy") {
      return { status: state.status, restarted: false };
    }

    assertRequiredEnv(this.containerEnv);
    await this.start();

    const started = await this.getState();
    console.log(`[WorkerContainer] restarted from status=${state.status}`);
    return { status: started.status, restarted: true };
  }

  async status(): Promise<{ status: string; exitCode?: number }> {
    const state = await this.getState();
    // exitCode only exists on the 'stopped_with_code' variant of State.
    return state.status === "stopped_with_code"
      ? { status: state.status, exitCode: state.exitCode }
      : { status: state.status };
  }

  async shutdown(): Promise<{ status: string }> {
    await this.stop();
    return { status: "stopping" };
  }
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json" },
  });

/**
 * Cold-start retry for requests proxied to the API container.
 *
 * containerFetch() does not throw on startup/proxy problems — it converts them
 * into plain-text error Responses (500 "Failed to start container: …",
 * 500 "Container suddenly disconnected…", 500 "Error proxying request to
 * container: …", 503 "There is no Container instance available…"). A request
 * that lands while the container is still booting therefore looks like one of
 * those responses, not an exception. Retry such responses exactly once after a
 * short delay; by then the container is normally healthy. Genuinely app-level
 * 4xx/5xx bodies do not match and are returned untouched.
 */
/** Length-independent comparison, so the token is not guessable by timing. */
function tokenMatches(provided: string | null, expected: string): boolean {
  if (!provided || provided.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < provided.length; i++) {
    diff |= provided.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return diff === 0;
}

/**
 * Adapter-only control surface, namespaced under /__cf/ so it cannot collide
 * with the app's routes (/api/v1/*, /ws/*, /health). Fails closed: with no
 * CF_ADMIN_TOKEN secret set, these routes are disabled entirely.
 */
async function handleAdmin(request: Request, workerEnv: Env, url: URL): Promise<Response> {
  const expected = workerEnv.CF_ADMIN_TOKEN;
  if (!expected) {
    return json({ error: "admin routes disabled: CF_ADMIN_TOKEN is not set" }, 503);
  }
  if (!tokenMatches(request.headers.get("x-cf-admin-token"), expected)) {
    return json({ error: "unauthorized" }, 401);
  }

  const worker = getContainer(workerEnv.WORKER_CONTAINER, WORKER_INSTANCE_NAME);
  console.log(`[cf] admin ${url.pathname} -> WORKER_CONTAINER binding acquired`);

  switch (url.pathname) {
    case "/__cf/status": {
      console.log(`[cf] WORKER_CONTAINER.status() before`);
      const status = await worker.status();
      console.log(`[cf] WORKER_CONTAINER.status() after -> ${JSON.stringify(status)}`);
      return json({
        worker: status,
        workerInstance: WORKER_INSTANCE_NAME,
        apiPoolSize: API_POOL_SIZE,
        apiPort: API_PORT,
      });
    }

    case "/__cf/worker/start": {
      console.log(`[cf] WORKER_CONTAINER.ensureRunning() before`);
      const result = await worker.ensureRunning();
      console.log(`[cf] WORKER_CONTAINER.ensureRunning() after -> ${JSON.stringify(result)}`);
      return json(result);
    }

    case "/__cf/worker/stop": {
      console.log(`[cf] WORKER_CONTAINER.shutdown() before`);
      const result = await worker.shutdown();
      console.log(`[cf] WORKER_CONTAINER.shutdown() after -> ${JSON.stringify(result)}`);
      return json(result);
    }

    default:
      return json({ error: "not found" }, 404);
  }
}

export default {
  /** Everything that is not an adapter route goes to the API container. */
  async fetch(request: Request, workerEnv: Env): Promise<Response> {
    const startedAt = Date.now();
    const requestedUrl = request.url;
    // Observe the request up front so a failing path is still attributable to
    // the method + URL even if everything below throws.
    console.log(`[cf] request ${request.method} ${requestedUrl}`);

    try {
      const url = new URL(requestedUrl);

      if (url.pathname.startsWith("/__cf/")) {
        return await handleAdmin(request, workerEnv, url);
      }

      // Stateless spread across the pool. Cross-instance fan-out for WebSockets
      // is handled by Redis pub/sub inside the app, so any instance can serve
      // any client. The stub's fetch() reaches ApiContainer.fetch(), which
      // auto-starts the container and proxies through containerFetch() — that
      // path accepts both WebSocketPair ends, so /ws/* upgrades still work.
      console.log(`[cf] getRandom(API_CONTAINER) before for ${request.method} ${url.pathname}`);
      const api = await getRandom(workerEnv.API_CONTAINER, API_POOL_SIZE);
      console.log(`[cf] getRandom(API_CONTAINER) after -> stub selected`);

      console.log(`[cf] API_CONTAINER.fetch() before for ${request.method} ${url.pathname}`);
      // The DO's ApiContainer.fetch() handles cold-start + retry internally, so
      // this is a straight proxy — no Worker-level retry wrapper needed.
      const res = await api.fetch(request);
      console.log(
        `[cf] API_CONTAINER.fetch() after -> status=${res.status} (${Date.now() - startedAt}ms) for ${request.method} ${url.pathname}`,
      );
      return res;
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      console.error(`[cf] unhandled exception for ${request.method} ${requestedUrl}`, {
        message: err.message,
        stack: err.stack,
        elapsedMs: Date.now() - startedAt,
      });
      return new Response(
        JSON.stringify({ error: err.message, stack: err.stack }),
        {
          status: 500,
          headers: { "content-type": "application/json" },
        },
      );
    }
  },

  /**
   * Supervisor tick (every 2 minutes, see `triggers.crons` in wrangler.json).
   * The worker keeps itself awake; this only recovers it after a
   * platform-initiated stop. 
   */
  async scheduled(
    _controller: ScheduledController,
    workerEnv: Env,
    ctx: ExecutionContext,
  ): Promise<void> {
    ctx.waitUntil(
      (async () => {
        try {
          console.log(`[cf] scheduled supervisor: WORKER_CONTAINER.ensureRunning() before`);
          const worker = getContainer(workerEnv.WORKER_CONTAINER, WORKER_INSTANCE_NAME);
          const result = await worker.ensureRunning();
          console.log(`[cf] scheduled supervisor: WORKER_CONTAINER.ensureRunning() after -> ${JSON.stringify(result)}`);
          if (result.restarted) {
            console.log(`[supervisor] worker restarted, status=${result.status}`);
          }
        } catch (e) {
          const err = e instanceof Error ? e : new Error(String(e));
          console.error(`[supervisor] WORKER_CONTAINER.ensureRunning() failed`, {
            message: err.message,
            stack: err.stack,
          });
        }
      })(),
    );
  },
};