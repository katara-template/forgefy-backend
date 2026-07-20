/** Auth value types and the helpers that parse GoTrue's responses. */

/** The authenticated user. `metadata` is GoTrue's `user_metadata`. */
export interface ForgefyUser {
  id: string;
  email?: string;
  metadata: Record<string, unknown>;
  /** The full user object from the backend, for fields not surfaced above. */
  raw: Record<string, unknown>;
}

/**
 * A signed-in session. `accessToken` is empty when a sign-up needs email
 * confirmation before a session is issued — check {@link isAuthenticated}.
 */
export interface ForgefySession {
  accessToken: string;
  refreshToken?: string;
  /** Absolute expiry, epoch seconds. Undefined when the backend omitted it. */
  expiresAt?: number;
  user?: ForgefyUser;
}

export type AuthChangeEvent = "signedIn" | "signedOut" | "tokenRefreshed";

export interface AuthChange {
  event: AuthChangeEvent;
  session: ForgefySession | null;
}

export function isAuthenticated(session: ForgefySession | null | undefined): boolean {
  return !!session && session.accessToken.length > 0;
}

/** True within `leewaySeconds` of expiry, so callers refresh a little early. */
export function isExpired(session: ForgefySession, leewaySeconds = 30): boolean {
  if (session.expiresAt === undefined) return false;
  const now = Math.floor(Date.now() / 1000);
  return now >= session.expiresAt - leewaySeconds;
}

type Json = Record<string, unknown>;

export function parseUser(json: Json): ForgefyUser {
  const metadata = json.user_metadata;
  return {
    id: String(json.id ?? ""),
    email: typeof json.email === "string" ? json.email : undefined,
    metadata: isObject(metadata) ? metadata : {},
    raw: json,
  };
}

export function parseSession(json: Json): ForgefySession {
  // GoTrue password grant → tokens at top level with a nested `user`.
  // GoTrue signup awaiting confirmation → the user object *is* the top level.
  const hasToken = typeof json.access_token === "string";
  const userJson = isObject(json.user) ? json.user : hasToken ? undefined : json;

  let expiresAt = typeof json.expires_at === "number" ? json.expires_at : undefined;
  if (expiresAt === undefined && typeof json.expires_in === "number") {
    expiresAt = Math.floor(Date.now() / 1000) + json.expires_in;
  }

  return {
    accessToken: typeof json.access_token === "string" ? json.access_token : "",
    refreshToken: typeof json.refresh_token === "string" ? json.refresh_token : undefined,
    expiresAt,
    user: userJson ? parseUser(userJson) : undefined,
  };
}

/** Serialise for a SessionStore (round-trips through parseSession). */
export function serializeSession(session: ForgefySession): string {
  const payload: Json = {
    access_token: session.accessToken,
    ...(session.refreshToken !== undefined ? { refresh_token: session.refreshToken } : {}),
    ...(session.expiresAt !== undefined ? { expires_at: session.expiresAt } : {}),
    ...(session.user ? { user: session.user.raw } : {}),
  };
  return JSON.stringify(payload);
}

function isObject(value: unknown): value is Json {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
