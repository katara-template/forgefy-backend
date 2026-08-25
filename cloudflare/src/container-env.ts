/**
 * Environment pass-through for the Cloudflare deployment adapter.
 *
 * The app's env var names are the contract. Nothing here renames, prefixes or
 * restructures them: whatever lands in the Worker's `env` (from `vars` in
 * wrangler.json or from `wrangler secret bulk`) is handed to the container
 * under exactly the same name, so ./Dockerfile and ./Dockerfile.worker stay
 * identical to what Docker Compose and other platforms run.
 *
 * This is a deny-list rather than an allow-list on purpose: adding a new
 * setting to .env and pushing it as a secret is all it takes: no edit here.
 */

/** Keys that belong to the adapter itself and must not leak into the app. */
const ADAPTER_ONLY_KEYS = new Set<string>(["CF_ADMIN_TOKEN"]);

/**
 * Without these the container starts and then fails in a way that is annoying
 * to diagnose (Celery retries a broker it cannot reach, forever). Better to
 * refuse to start.
 */
const REQUIRED_KEYS = [
  "REDIS_URL",
  "CELERY_BROKER_URL",
  "CELERY_RESULT_BACKEND",
  "SECRET_KEY",
] as const;

/**
 * Project every string-ish value in the Worker env into a plain container
 * environment. Bindings (Durable Object namespaces, KV, R2, ...) are objects
 * and are skipped.
 */
export function buildContainerEnv(
  workerEnv: Record<string, unknown>,
): Record<string, string> {
  const containerEnv: Record<string, string> = {};

  for (const [key, value] of Object.entries(workerEnv)) {
    if (ADAPTER_ONLY_KEYS.has(key)) continue;

    if (typeof value === "string") {
      containerEnv[key] = value;
    } else if (typeof value === "number" || typeof value === "boolean") {
      // wrangler.json `vars` may hold JSON numbers/booleans; the container
      // expects strings, same as a shell would provide.
      containerEnv[key] = String(value);
    }
  }

  return containerEnv;
}

/** Throws with every missing key at once, rather than one deploy at a time. */
export function assertRequiredEnv(containerEnv: Record<string, string>): void {
  const missing = REQUIRED_KEYS.filter((key) => !containerEnv[key]);

  if (missing.length > 0) {
    throw new Error(
      `Missing required environment variable(s): ${missing.join(", ")}. ` +
        `Upload them with: wrangler secret bulk ../.env -c wrangler.json`,
    );
  }
}

/**
 * Stable name for the single Celery worker instance. Using a fixed name means
 * every cron tick, every admin call and every deploy addresses the same
 * Durable Object, and therefore the same container.
 */
export const WORKER_INSTANCE_NAME = "celery-primary";

/** Port uvicorn listens on inside ./Dockerfile (EXPOSE 8000, `${PORT:-8000}`). */
export const API_PORT = 8000;

/** Must not exceed `max_instances` for ApiContainer in wrangler.json. */
export const API_POOL_SIZE = 5;
