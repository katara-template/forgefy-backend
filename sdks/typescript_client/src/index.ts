/**
 * Forgefy app-runtime SDK for Next.js & React Native.
 *
 * The auth, data, and networking layer that Forgefy-generated apps depend on,
 * so the build agent wires in a tested primitive instead of regenerating a
 * data/service layer on every build. Universal `fetch` — runs in Next.js route
 * handlers and client components, and in React Native.
 */

export { ForgefyClient } from "./client.js";
export {
  type ForgefyClientOptions,
  type ForgefyProvider,
  resolveConfig,
} from "./config.js";

export { ForgefyAuth } from "./auth/auth.js";
export type { SignInParams, SignUpParams, Subscription } from "./auth/auth.js";
export {
  type AuthChange,
  type AuthChangeEvent,
  type ForgefySession,
  type ForgefyUser,
  isAuthenticated,
  isExpired,
} from "./auth/types.js";
export {
  InMemorySessionStore,
  type KeyValueStorage,
  persistentSessionStore,
  type SessionStore,
} from "./auth/session-store.js";

export { ForgefyQuery } from "./data/query.js";

export { type ForgefyResponse } from "./http.js";

export {
  AuthError,
  ConflictError,
  ConnectionError,
  ForgefyError,
  type ForgefyErrorOptions,
  NotFoundError,
  RateLimitError,
  ServerError,
  ValidationError,
} from "./errors.js";
