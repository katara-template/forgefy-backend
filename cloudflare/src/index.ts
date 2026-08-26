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
import { env } from "cloudflare:workers";

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

/**
 * Built once per isolate from the Worker env. Contains REDIS_URL,
 * CELERY_BROKER_URL, every API key — under their original names.
 */
const CONTAINER_ENV = buildContainerEnv(env as unknown as Record<string, unknown>);

// ---------------------------------------------------------------------------
// API container
// ---------------------------------------------------------------------------

export class ApiContainer extends Container<Env> {
  defaultPort = API_PORT;

  /**
   * The API is allowed to scale to zero. Note the trade-off for WebSockets:
   * a socket held open with no new HTTP requests for an hour can be dropped
   * when the instance sleeps. Clients should reconnect; the WS handlers
   * rebuild state from Redis pub/sub anyway (app/api/ws/connection_manager.py).
   */
  sleepAfter = "1h";

  envVars = CONTAINER_ENV;

  private booted = false;

  override onStart(): void {
    console.log(`[ApiContainer] uvicorn listening on :${API_PORT}`);
  }

  override onStop({ exitCode, reason }: { exitCode: number; reason: string }): void {
    console.log(`[ApiContainer] stopped exitCode=${exitCode} reason=${reason}`);
  }

  /**
   * The default port-ready window is 20s. This image is ~8.23 GB and the app
   * imports playwright, firebase and the AI SDKs at module scope, so a cold
   * start can run past that. Boot explicitly with a longer window, once, then
   * hand off to the base implementation (which preserves WebSocket upgrades).
   */
  override async fetch(request: Request): Promise<Response> {
    if (!this.booted) {
      await this.ctx.blockConcurrencyWhile(async () => {
        if (this.booted) return;
        await this.startAndWaitForPorts({
          ports: [API_PORT],
          cancellationOptions: { portReadyTimeoutMS: 60_000 },
        });
        this.booted = true;
      });
    }

    return super.fetch(request);
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

  envVars = CONTAINER_ENV;

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

    assertRequiredEnv(CONTAINER_ENV);
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

  switch (url.pathname) {
    case "/__cf/status":
      return json({
        worker: await worker.status(),
        workerInstance: WORKER_INSTANCE_NAME,
        apiPoolSize: API_POOL_SIZE,
        apiPort: API_PORT,
      });

    case "/__cf/worker/start":
      return json(await worker.ensureRunning());

    case "/__cf/worker/stop":
      return json(await worker.shutdown());

    default:
      return json({ error: "not found" }, 404);
  }
}

export default {
  /** Everything that is not an adapter route goes to the API container. */
  async fetch(request: Request, workerEnv: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/__cf/")) {
      return handleAdmin(request, workerEnv, url);
    }

    // Stateless spread across the pool. Cross-instance fan-out for WebSockets
    // is handled by Redis pub/sub inside the app, so any instance can serve
    // any client. Uses the stub's fetch() — containerFetch() would break the
    // WebSocket upgrades on /ws/*.
    const api = await getRandom(workerEnv.API_CONTAINER, API_POOL_SIZE);
    return api.fetch(request);
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
        const worker = getContainer(workerEnv.WORKER_CONTAINER, WORKER_INSTANCE_NAME);
        const result = await worker.ensureRunning();
        if (result.restarted) {
          console.log(`[supervisor] worker restarted, status=${result.status}`);
        }
      })(),
    );
  },
};
