# Changelog

## 0.1.0

Initial release — the app-runtime foundation the build agent wires into
generated apps.

- **Networking**: `ForgefyHttp` with retry + exponential backoff, `Retry-After`
  handling, timeouts, and a normalized `ForgefyException` hierarchy mapped from
  GoTrue / PostgREST / RFC 7807 error bodies.
- **Auth** (Supabase GoTrue): sign up / in / out, refresh, session persistence
  via a pluggable `SessionStore`, and an `onAuthStateChange` stream.
- **Data** (Supabase & Neon PostgREST): `from(table)` query builder with
  select / insert / update / delete / upsert, filters, `order`, `limit`,
  `range`, and `single`.
