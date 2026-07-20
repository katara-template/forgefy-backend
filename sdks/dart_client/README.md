# forgefy_client

The **app-runtime SDK** for Forgefy-generated Dart & Flutter apps — the auth,
data, and networking layer the build agent wires in instead of regenerating a
`core/` layer on every build.

> This is distinct from the [`forgefy`](../dart) package, which is the
> *developer* SDK for the meeting-extraction API. This package is what the
> **generated apps themselves** depend on.

It depends only on `package:http` (never `dart:io`), so the same code runs on
the Dart VM, Flutter mobile/desktop, and Flutter web.

## What it does today

| Layer | Backend | Surface |
|---|---|---|
| **Networking** | any | Retries with backoff, `Retry-After`, timeouts, one normalized error hierarchy |
| **Auth** | Supabase (GoTrue) | `signUp` · `signInWithPassword` · `signOut` · `refreshSession` · `restoreSession` · `onAuthStateChange` |
| **Data** | Supabase / Neon (PostgREST) | `from(table).select/insert/update/delete/upsert` + filters, `order`, `limit`, `range`, `single` |

Firebase and storage/realtime land later behind these same interfaces.

## Quick start

```dart
import 'package:forgefy_client/forgefy_client.dart';

final client = ForgefyClient(
  ForgefyConfig(
    url: 'https://your-project.supabase.co',
    anonKey: 'your-anon-key',
  ),
);

await client.auth.restoreSession();               // resume a stored login
await client.auth.signInWithPassword(
  email: 'a@b.c', password: 'pw',
);

// The signed-in user's token is attached to every query automatically.
final todos = await client
    .from('todos')
    .select('id, title, done')
    .eq('done', false)
    .order('created_at', ascending: false)
    .limit(20);
```

## Configuration

```dart
ForgefyConfig(
  url: '...',                       // Supabase URL or Neon Data API URL
  anonKey: '...',                   // public-safe key; access enforced by RLS
  provider: ForgefyProvider.supabase, // .neon for a Neon Data API
  maxRetries: 2,
  timeout: Duration(seconds: 30),
)
```

Only public-safe values ever go here — security is enforced by Row Level
Security / Postgres grants on the server, never by keeping the key secret. This
matches the contract the build agent already documents for generated apps.

### Staying signed in (Flutter)

The default session store is in-memory. In a Flutter app, pass a persistent
`SessionStore` so users stay logged in across restarts:

```dart
class PrefsSessionStore implements SessionStore {
  PrefsSessionStore(this._prefs);
  final SharedPreferences _prefs;
  static const _k = 'forgefy.session';

  @override
  Future<String?> read() async => _prefs.getString(_k);
  @override
  Future<void> write(String value) => _prefs.setString(_k, value);
  @override
  Future<void> delete() => _prefs.remove(_k);
}

final client = ForgefyClient(config, sessionStore: PrefsSessionStore(prefs));
```

## Errors

Every failure is a `ForgefyException` subclass, mapped from the backend's status
regardless of which error envelope it used (GoTrue, PostgREST, or RFC 7807):

`ValidationException` (400/422) · `AuthException` (401/403) ·
`NotFoundException` (404) · `ConflictException` (409) ·
`RateLimitException` (429) · `ServerException` (5xx) · `ConnectionException`
(no response). Each carries `status`, and — where the backend supplies them —
`code` and `details`.

## How the build agent uses it

In the generated Clean-Architecture layout, `*_remote_datasource.dart` calls
this SDK instead of a hand-rolled `api_client.dart` + provider SDK:

```dart
class TodoRemoteDataSource {
  TodoRemoteDataSource(this._client);
  final ForgefyClient _client;

  Future<List<TodoModel>> fetchOpen(String userId) async {
    final result = await _client
        .from('todos')
        .select()
        .eq('user_id', userId)
        .eq('done', false)
        .order('created_at', ascending: false);
    return (result as List<dynamic>)
        .map((r) => TodoModel.fromJson(r as Map<String, dynamic>))
        .toList();
  }
}
```

## Development

```bash
dart pub get
dart analyze
dart test
```
