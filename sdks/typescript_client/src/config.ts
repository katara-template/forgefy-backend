/** Runtime configuration for a Forgefy-generated app's backend. */

import type { SessionStore } from "./auth/session-store.js";

/**
 * The backend a generated app was wired to at build time. Both `supabase` and
 * `neon` speak PostgREST for data (so they share the data layer); only
 * `supabase` provides auth today. `firebase` is reserved.
 */
export type ForgefyProvider = "supabase" | "neon" | "firebase";

export interface ForgefyClientOptions {
  /** Project root, e.g. "https://abcd.supabase.co", or the Neon Data API URL. */
  url: string;
  /**
   * Public API key (Supabase anon key / Neon publishable key). Only ever a
   * public-safe value — access is enforced by Row Level Security / Postgres
   * grants on the server, never by keeping the key secret.
   */
  anonKey: string;
  /** Defaults to "supabase". */
  provider?: ForgefyProvider;
  /** Retries for rate-limited/transient failures. Default 2. */
  maxRetries?: number;
  /** Per-request timeout. Default 30s. */
  timeoutMs?: number;
  /** Base backoff between retries. Exposed for tests; leave default otherwise. */
  retryDelayMs?: number;
  /**
   * A user access token to act as, when there is no interactive sign-in — the
   * common case in a Next.js route handler, where the JWT arrives on the
   * request. Queries run as that user (RLS applies). A session established via
   * {@link ForgefyAuth} takes precedence over this.
   */
  accessToken?: string;
  /** Injectable fetch (tests, or Node <18). Defaults to globalThis.fetch. */
  fetch?: typeof fetch;
  /**
   * Where the signed-in session is persisted. Defaults to in-memory. Pass a
   * persistent store (localStorage on web, AsyncStorage on React Native — see
   * {@link persistentSessionStore}) to keep users logged in across restarts.
   */
  sessionStore?: SessionStore;
}

/** Normalised, fully-defaulted config used internally. */
export interface ResolvedConfig {
  url: string;
  anonKey: string;
  provider: ForgefyProvider;
  authUrl: string;
  restUrl: string;
  sendsApiKeyHeader: boolean;
  maxRetries: number;
  timeoutMs: number;
  retryDelayMs: number;
  fetch: typeof fetch;
}

export function resolveConfig(options: ForgefyClientOptions): ResolvedConfig {
  if (!options.url) throw new Error("ForgefyClient: url is required");
  if (!options.anonKey) throw new Error("ForgefyClient: anonKey is required");

  const url = options.url.replace(/\/+$/, "");
  const provider = options.provider ?? "supabase";

  // Bind to the global object: browsers throw "Illegal invocation" if
  // globalThis.fetch is called with any other `this`. A caller-supplied fetch
  // is used as-is — they own its binding.
  const resolvedFetch = options.fetch ?? globalThis.fetch;
  if (!resolvedFetch) {
    throw new Error(
      "ForgefyClient: no fetch available — pass options.fetch (Node <18 or a non-fetch environment)",
    );
  }

  return {
    url,
    anonKey: options.anonKey,
    provider,
    authUrl: `${url}/auth/v1`,
    // Supabase namespaces PostgREST under /rest/v1; a Neon Data API URL is
    // already the PostgREST root.
    restUrl: provider === "supabase" ? `${url}/rest/v1` : url,
    sendsApiKeyHeader: provider === "supabase",
    maxRetries: options.maxRetries ?? 2,
    timeoutMs: options.timeoutMs ?? 30_000,
    retryDelayMs: options.retryDelayMs ?? 500,
    fetch: options.fetch ? resolvedFetch : resolvedFetch.bind(globalThis),
  };
}
