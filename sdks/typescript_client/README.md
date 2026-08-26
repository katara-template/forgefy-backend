# @forgefy/client

The **app-runtime SDK** for Forgefy-generated **Next.js** and **React Native**
apps — the auth, data, and networking layer the build agent wires in instead of
regenerating a `lib/services` / datasource layer on every build.

> This is distinct from [`@forgefy/sdk`](../typescript), the *developer* SDK for
> the meeting-extraction API. This package is what the **generated apps
> themselves** depend on. It's the TypeScript sibling of the Flutter
> [`forgefy_client`](../dart_client) package.

Zero runtime dependencies — it uses the platform `fetch`, so the same code runs
in Next.js route handlers (Node), client components (browser), and React Native.

## What it does today

| Layer | Backend | Surface |
|---|---|---|
| **Networking** | any | Retries with backoff, `Retry-After`, `AbortController` timeouts, one normalized error hierarchy |
| **Auth** | Supabase (GoTrue) | `signUp` · `signInWithPassword` · `signOut` · `refreshSession` · `restoreSession` · `onAuthStateChange` |
| **Data** | Supabase / Neon (PostgREST) | `from(table).select/insert/update/delete/upsert` + filters, `order`, `limit`, `range`, `single` |

## Install

```bash
npm install @forgefy/client
```

## Quick start

```ts
import { ForgefyClient } from "@forgefy/client";

const client = new ForgefyClient({
  url: process.env.NEXT_PUBLIC_SUPABASE_URL!,
  anonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
});

await client.auth.signInWithPassword({ email, password });

// The signed-in user's token is attached to every query automatically.
type Todo = { id: string; title: string; done: boolean };
const open = await client
  .from<Todo>("todos")
  .select("id, title, done")
  .eq("done", false)
  .order("created_at", { ascending: false })
  .limit(20);
```

## React Native

Persist the session with AsyncStorage so users stay signed in across launches:

```ts
import AsyncStorage from "@react-native-async-storage/async-storage";
import { ForgefyClient, persistentSessionStore } from "@forgefy/client";

export const forgefy = new ForgefyClient({
  url: process.env.EXPO_PUBLIC_SUPABASE_URL!,
  anonKey: process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY!,
  sessionStore: persistentSessionStore(AsyncStorage),
});

// On app start:
await forgefy.auth.restoreSession();
```

`persistentSessionStore` also accepts `window.localStorage` in a Next.js client
component.

## Next.js

The build agent's rule is that anything touching the database runs on the
server. Create one client per request inside a route handler, seeding it with
the caller's JWT so queries run **as that user** (RLS applies):

```ts
// app/api/todos/route.ts
import { ForgefyClient } from "@forgefy/client";

export async function GET(req: Request) {
  const token = req.headers.get("authorization")?.replace("Bearer ", "");
  const client = new ForgefyClient({
    url: process.env.NEXT_PUBLIC_SUPABASE_URL!,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    accessToken: token, // queries run as this user
  });
  const todos = await client.from("todos").select().eq("done", false);
  return Response.json(todos);
}
```

Client components call these route handlers via `fetch` — they never import the
SDK directly, matching the generated-app architecture.

## Configuration

```ts
new ForgefyClient({
  url: "...",             // Supabase URL or Neon Data API URL
  anonKey: "...",         // public-safe key; access enforced by RLS
  provider: "supabase",   // or "neon"
  maxRetries: 2,
  timeoutMs: 30_000,
  sessionStore,           // default: in-memory
  fetch,                  // default: globalThis.fetch
});
```

Only public-safe values go here — security is enforced by Row Level Security /
Postgres grants on the server, never by keeping the key secret.

## Errors

Every failure is a `ForgefyError` subclass, mapped from the backend's status
regardless of its error envelope (GoTrue, PostgREST, or RFC 7807):

`ValidationError` (400/422) · `AuthError` (401/403) · `NotFoundError` (404) ·
`ConflictError` (409) · `RateLimitError` (429) · `ServerError` (5xx) ·
`ConnectionError` (no response). Each carries `status`, and — where the backend
supplies them — `code` and `details`.

```ts
import { ConflictError } from "@forgefy/client";

try {
  await client.from("users").insert({ email });
} catch (e) {
  if (e instanceof ConflictError) {
    // e.code === "23505" (unique violation)
  }
}
```

## Development

```bash
npm install
npm run typecheck
npm test
npm run build
```
