# Changelog

## 0.1.0

Initial release — the app-runtime foundation the build agent wires into
generated Next.js and React Native apps. TypeScript sibling of the Flutter
`forgefy_client` package.

- **Networking**: `ForgefyHttp` over universal `fetch` with retry + exponential
  backoff, `Retry-After` handling, `AbortController` timeouts, and a normalized
  `ForgefyError` hierarchy mapped from GoTrue / PostgREST / RFC 7807 bodies.
- **Auth** (Supabase GoTrue): sign up / in / out, refresh, session persistence
  via a pluggable `SessionStore` (`persistentSessionStore` adapts `localStorage`
  and React Native `AsyncStorage`), and `onAuthStateChange`.
- **Data** (Supabase & Neon PostgREST): a typed `from<T>(table)` query builder
  (a thenable) with select / insert / update / delete / upsert, filters,
  `order`, `limit`, `range`, and `single`.
- `accessToken` option to run queries as a specific user in a Next.js route
  handler without an interactive sign-in.
