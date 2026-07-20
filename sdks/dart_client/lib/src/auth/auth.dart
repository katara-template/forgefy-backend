/// Authentication against Supabase GoTrue.
///
/// One concrete backend today; the surface (sign up / in / out, refresh,
/// session recovery, an auth-state stream) is provider-neutral so a Firebase
/// implementation can land behind the same API later.
library;

import 'dart:async';
import 'dart:convert';

import '../config.dart';
import '../errors.dart';
import '../http.dart';
import 'models.dart';
import 'session_store.dart';

/// Managed via [ForgefyClient.auth]. Holds the current session and keeps the
/// shared transport's bearer token in sync as it changes.
class ForgefyAuth {
  ForgefyAuth(this._http, this._config, this._store);

  final ForgefyHttp _http;
  final ForgefyConfig _config;
  final SessionStore _store;

  final StreamController<AuthChange> _changes =
      StreamController<AuthChange>.broadcast();

  ForgefySession? _session;

  /// The live session, or null when signed out.
  ForgefySession? get currentSession => _session;

  /// The signed-in user, or null when signed out.
  ForgefyUser? get currentUser => _session?.user;

  /// Emits on every sign-in, sign-out, and token refresh. `broadcast`, so it is
  /// safe to listen from multiple widgets.
  Stream<AuthChange> get onAuthStateChange => _changes.stream;

  /// The token [ForgefyHttp] injects. Exposed so [ForgefyClient] can wire it.
  String? get accessToken => _session?.accessToken;

  /// Restore a persisted session on app start, refreshing it if it has expired.
  /// Call once before the first authed request; safe to call when nothing is
  /// stored (returns null).
  Future<ForgefySession?> restoreSession() async {
    final stored = await _store.read();
    if (stored == null || stored.isEmpty) return null;

    ForgefySession session;
    try {
      session = ForgefySession.fromJson(jsonDecode(stored) as Map<String, dynamic>);
    } catch (_) {
      await _store.delete();
      return null;
    }

    if (session.isExpired() && session.refreshToken != null) {
      try {
        return await refreshSession(session.refreshToken!);
      } catch (_) {
        await signOut();
        return null;
      }
    }
    _session = session;
    return session;
  }

  /// Create an account. Returns a session with tokens when the project
  /// auto-confirms; otherwise the returned session is unauthenticated (email
  /// confirmation pending) and no auth-state event fires.
  Future<ForgefySession> signUp({
    required String email,
    required String password,
    Map<String, dynamic>? data,
  }) async {
    _requireSupabase();
    final res = await _http.send(
      'POST',
      '${_config.authUrl}/signup',
      body: {
        'email': email,
        'password': password,
        if (data != null) 'data': data,
      },
    );
    final session = ForgefySession.fromJson(_asMap(res.data));
    if (session.isAuthenticated) {
      await _persist(session, AuthChangeEvent.signedIn);
    }
    return session;
  }

  /// Sign in with email + password.
  Future<ForgefySession> signInWithPassword({
    required String email,
    required String password,
  }) async {
    _requireSupabase();
    final res = await _http.send(
      'POST',
      '${_config.authUrl}/token?grant_type=password',
      body: {'email': email, 'password': password},
    );
    final session = ForgefySession.fromJson(_asMap(res.data));
    await _persist(session, AuthChangeEvent.signedIn);
    return session;
  }

  /// Exchange a refresh token for a fresh session. Usually automatic via
  /// [restoreSession]; exposed for callers that manage refresh themselves.
  Future<ForgefySession> refreshSession(String refreshToken) async {
    _requireSupabase();
    final res = await _http.send(
      'POST',
      '${_config.authUrl}/token?grant_type=refresh_token',
      body: {'refresh_token': refreshToken},
      retryOn5xx: true, // idempotent
    );
    final session = ForgefySession.fromJson(_asMap(res.data));
    await _persist(session, AuthChangeEvent.tokenRefreshed);
    return session;
  }

  /// Sign out: revoke the session server-side (best-effort) and clear it
  /// locally. Local state is always cleared even if the network call fails.
  Future<void> signOut() async {
    final token = _session?.accessToken;
    if (token != null && token.isNotEmpty) {
      try {
        await _http.send('POST', '${_config.authUrl}/logout');
      } on ForgefyException {
        // The token may already be invalid; clearing locally is what matters.
      }
    }
    _session = null;
    await _store.delete();
    _changes.add(AuthChange(AuthChangeEvent.signedOut, null));
  }

  /// Release the auth-state stream. Called by [ForgefyClient.close].
  Future<void> close() => _changes.close();

  Future<void> _persist(ForgefySession session, AuthChangeEvent event) async {
    _session = session;
    await _store.write(jsonEncode(session.toJson()));
    _changes.add(AuthChange(event, session));
  }

  void _requireSupabase() {
    if (_config.provider != ForgefyProvider.supabase) {
      throw StateError(
        'ForgefyAuth supports the Supabase provider only; '
        'this client is configured for ${_config.provider.name}.',
      );
    }
  }

  Map<String, dynamic> _asMap(Object? data) =>
      data is Map<String, dynamic> ? data : <String, dynamic>{};
}
