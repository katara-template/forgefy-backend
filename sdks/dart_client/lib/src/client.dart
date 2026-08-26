/// The entry point a Forgefy-generated app talks to.
library;

import 'package:http/http.dart' as http;

import 'auth/auth.dart';
import 'auth/session_store.dart';
import 'config.dart';
import 'data/query.dart';
import 'http.dart';

/// Wires the shared transport to [auth] and the data layer so a signed-in
/// user's token flows to every query automatically.
///
/// ```dart
/// final client = ForgefyClient(
///   ForgefyConfig(url: 'https://xyz.supabase.co', anonKey: '...'),
/// );
/// await client.auth.restoreSession();
/// await client.auth.signInWithPassword(email: e, password: p);
/// final todos = await client.from('todos').select().eq('done', false);
/// ```
class ForgefyClient {
  ForgefyClient(
    this.config, {
    SessionStore? sessionStore,
    http.Client? httpClient,
  }) : _http = ForgefyHttp(config, httpClient: httpClient) {
    auth = ForgefyAuth(_http, config, sessionStore ?? InMemorySessionStore());
    // Every request bears the signed-in user's token, falling back to the anon
    // key (handled inside ForgefyHttp) before sign-in.
    _http.tokenProvider = () => auth.accessToken;
  }

  final ForgefyConfig config;
  final ForgefyHttp _http;

  /// Authentication and the current session.
  late final ForgefyAuth auth;

  /// Start a query against [table].
  ForgefyQuery from(String table) =>
      ForgefyQuery(_http, config.restUrl, table);

  /// Release the auth stream and the owned HTTP client. Call on app shutdown.
  Future<void> close() async {
    await auth.close();
    _http.close();
  }
}
