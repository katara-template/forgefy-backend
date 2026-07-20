/**
 * Authentication against Supabase GoTrue.
 *
 * One concrete backend today; the surface (sign up / in / out, refresh, session
 * recovery, an auth-state subscription) is provider-neutral so a Firebase
 * implementation can land behind the same API later.
 */

import type { ResolvedConfig } from "../config.js";
import { ForgefyError } from "../errors.js";
import type { ForgefyHttp } from "../http.js";
import type { SessionStore } from "./session-store.js";
import {
  type AuthChange,
  type AuthChangeEvent,
  type ForgefySession,
  type ForgefyUser,
  isAuthenticated,
  isExpired,
  parseSession,
  serializeSession,
} from "./types.js";

export interface SignUpParams {
  email: string;
  password: string;
  /** Extra fields stored on the user (GoTrue `data` → `user_metadata`). */
  data?: Record<string, unknown>;
}

export interface SignInParams {
  email: string;
  password: string;
}

/** Unsubscribe handle returned by {@link ForgefyAuth.onAuthStateChange}. */
export interface Subscription {
  unsubscribe(): void;
}

/** Managed via `ForgefyClient.auth`. Holds the current session. */
export class ForgefyAuth {
  private session: ForgefySession | null = null;
  private readonly listeners = new Set<(change: AuthChange) => void>();

  constructor(
    private readonly http: ForgefyHttp,
    private readonly cfg: ResolvedConfig,
    private readonly store: SessionStore,
  ) {}

  /** The live session, or null when signed out. */
  get currentSession(): ForgefySession | null {
    return this.session;
  }

  /** The signed-in user, or null when signed out. */
  get currentUser(): ForgefyUser | null {
    return this.session?.user ?? null;
  }

  /** The token ForgefyHttp injects. Used by ForgefyClient to wire the transport. */
  get accessToken(): string | null {
    return this.session?.accessToken ?? null;
  }

  /** Subscribe to sign-in / sign-out / token-refresh events. */
  onAuthStateChange(callback: (change: AuthChange) => void): Subscription {
    this.listeners.add(callback);
    return { unsubscribe: () => this.listeners.delete(callback) };
  }

  /**
   * Restore a persisted session on app start, refreshing it if it has expired.
   * Call once before the first authed request; safe when nothing is stored.
   */
  async restoreSession(): Promise<ForgefySession | null> {
    const stored = await this.store.read();
    if (!stored) return null;

    let session: ForgefySession;
    try {
      session = parseSession(JSON.parse(stored) as Record<string, unknown>);
    } catch {
      await this.store.delete();
      return null;
    }

    if (isExpired(session) && session.refreshToken) {
      try {
        return await this.refreshSession(session.refreshToken);
      } catch {
        await this.signOut();
        return null;
      }
    }
    this.session = session;
    return session;
  }

  /**
   * Create an account. Returns a session with tokens when the project
   * auto-confirms; otherwise the returned session is unauthenticated (email
   * confirmation pending) and no auth-state event fires.
   */
  async signUp(params: SignUpParams): Promise<ForgefySession> {
    this.requireSupabase();
    const res = await this.http.send("POST", `${this.cfg.authUrl}/signup`, {
      body: {
        email: params.email,
        password: params.password,
        ...(params.data !== undefined ? { data: params.data } : {}),
      },
    });
    const session = parseSession(asRecord(res.data));
    if (isAuthenticated(session)) await this.persist(session, "signedIn");
    return session;
  }

  /** Sign in with email + password. */
  async signInWithPassword(params: SignInParams): Promise<ForgefySession> {
    this.requireSupabase();
    const res = await this.http.send("POST", `${this.cfg.authUrl}/token?grant_type=password`, {
      body: { email: params.email, password: params.password },
    });
    const session = parseSession(asRecord(res.data));
    await this.persist(session, "signedIn");
    return session;
  }

  /**
   * Exchange a refresh token for a fresh session. Usually automatic via
   * {@link restoreSession}; exposed for callers managing refresh themselves.
   */
  async refreshSession(refreshToken: string): Promise<ForgefySession> {
    this.requireSupabase();
    const res = await this.http.send("POST", `${this.cfg.authUrl}/token?grant_type=refresh_token`, {
      body: { refresh_token: refreshToken },
      retryOn5xx: true, // idempotent
    });
    const session = parseSession(asRecord(res.data));
    await this.persist(session, "tokenRefreshed");
    return session;
  }

  /**
   * Sign out: revoke the session server-side (best-effort) and clear it
   * locally. Local state is cleared even if the network call fails.
   */
  async signOut(): Promise<void> {
    const token = this.session?.accessToken;
    if (token) {
      try {
        await this.http.send("POST", `${this.cfg.authUrl}/logout`);
      } catch (err) {
        if (!(err instanceof ForgefyError)) throw err;
        // The token may already be invalid; clearing locally is what matters.
      }
    }
    this.session = null;
    await this.store.delete();
    this.emit({ event: "signedOut", session: null });
  }

  private async persist(session: ForgefySession, event: AuthChangeEvent): Promise<void> {
    this.session = session;
    await this.store.write(serializeSession(session));
    this.emit({ event, session });
  }

  private emit(change: AuthChange): void {
    for (const listener of this.listeners) listener(change);
  }

  private requireSupabase(): void {
    if (this.cfg.provider !== "supabase") {
      throw new Error(
        `ForgefyAuth supports the Supabase provider only; this client is configured for ${this.cfg.provider}.`,
      );
    }
  }
}

function asRecord(data: unknown): Record<string, unknown> {
  return data !== null && typeof data === "object" ? (data as Record<string, unknown>) : {};
}
