/** The entry point a Forgefy-generated app talks to. */

import { ForgefyAuth } from "./auth/auth.js";
import { InMemorySessionStore } from "./auth/session-store.js";
import { type ForgefyClientOptions, resolveConfig } from "./config.js";
import { ForgefyQuery } from "./data/query.js";
import { ForgefyHttp } from "./http.js";

type Row = Record<string, unknown>;

/**
 * Wires the shared transport to {@link auth} and the data layer so a signed-in
 * user's token flows to every query automatically.
 *
 * ```ts
 * const client = new ForgefyClient({
 *   url: "https://xyz.supabase.co",
 *   anonKey: "...",
 * });
 * await client.auth.restoreSession();
 * await client.auth.signInWithPassword({ email, password });
 * const todos = await client.from("todos").select().eq("done", false);
 * ```
 */
export class ForgefyClient {
  readonly auth: ForgefyAuth;

  private readonly http: ForgefyHttp;
  private readonly restUrl: string;

  constructor(options: ForgefyClientOptions) {
    const cfg = resolveConfig(options);
    this.restUrl = cfg.restUrl;
    this.http = new ForgefyHttp(cfg);
    this.auth = new ForgefyAuth(this.http, cfg, options.sessionStore ?? new InMemorySessionStore());
    // Every request bears the signed-in user's token, then a server-supplied
    // access token (route-handler case), then falls back to the anon key
    // (handled inside ForgefyHttp).
    const seedToken = options.accessToken ?? null;
    this.http.tokenProvider = () => this.auth.accessToken ?? seedToken;
  }

  /** Start a query against `table`. Pass a row type for typed results. */
  from<T extends Row = Row>(table: string): ForgefyQuery<T> {
    return new ForgefyQuery<T>(this.http, this.restUrl, table);
  }
}
