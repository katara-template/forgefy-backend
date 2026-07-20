/// Auth value types: the signed-in user and the token-bearing session.
library;

/// The authenticated user. [metadata] is the provider's `user_metadata` — the
/// place custom profile fields (display name, avatar, …) live.
class ForgefyUser {
  ForgefyUser({
    required this.id,
    this.email,
    this.metadata = const {},
    this.raw = const {},
  });

  factory ForgefyUser.fromJson(Map<String, dynamic> json) {
    final metadata = json['user_metadata'];
    return ForgefyUser(
      id: (json['id'] ?? '').toString(),
      email: json['email'] as String?,
      metadata: metadata is Map<String, dynamic> ? metadata : const {},
      raw: json,
    );
  }

  final String id;
  final String? email;
  final Map<String, dynamic> metadata;

  /// The full user object as returned by the backend, for fields not surfaced
  /// as typed getters.
  final Map<String, dynamic> raw;

  Map<String, dynamic> toJson() => raw.isNotEmpty
      ? raw
      : {'id': id, 'email': email, 'user_metadata': metadata};
}

/// A signed-in session. [accessToken] is empty when a sign-up needs email
/// confirmation before a session is issued — check [isAuthenticated].
class ForgefySession {
  ForgefySession({
    required this.accessToken,
    this.refreshToken,
    this.expiresAt,
    this.user,
  });

  factory ForgefySession.fromJson(Map<String, dynamic> json) {
    // GoTrue password grant → tokens at top level with a nested `user`.
    // GoTrue signup awaiting confirmation → the user object *is* the top level.
    final hasToken = json['access_token'] is String;
    final userMap = json['user'];
    final userJson = userMap is Map<String, dynamic>
        ? userMap
        : (hasToken ? null : json);

    var expiresAt = json['expires_at'] as int?;
    if (expiresAt == null && json['expires_in'] is num) {
      expiresAt = DateTime.now().millisecondsSinceEpoch ~/ 1000 +
          (json['expires_in'] as num).toInt();
    }

    return ForgefySession(
      accessToken: (json['access_token'] as String?) ?? '',
      refreshToken: json['refresh_token'] as String?,
      expiresAt: expiresAt,
      user: userJson == null ? null : ForgefyUser.fromJson(userJson),
    );
  }

  final String accessToken;
  final String? refreshToken;

  /// Absolute expiry, epoch seconds. Null when the backend didn't provide one.
  final int? expiresAt;
  final ForgefyUser? user;

  bool get isAuthenticated => accessToken.isNotEmpty;

  /// True within [leeway] of expiry, so callers refresh a little early.
  bool isExpired({Duration leeway = const Duration(seconds: 30)}) {
    if (expiresAt == null) return false;
    final now = DateTime.now().millisecondsSinceEpoch ~/ 1000;
    return now >= expiresAt! - leeway.inSeconds;
  }

  Map<String, dynamic> toJson() => {
        'access_token': accessToken,
        if (refreshToken != null) 'refresh_token': refreshToken,
        if (expiresAt != null) 'expires_at': expiresAt,
        if (user != null) 'user': user!.toJson(),
      };
}

/// What happened to the session, delivered on [ForgefyAuth.onAuthStateChange].
enum AuthChangeEvent { signedIn, signedOut, tokenRefreshed }

/// An auth-state change: the [event] and the resulting [session] (null on
/// sign-out).
class AuthChange {
  AuthChange(this.event, this.session);

  final AuthChangeEvent event;
  final ForgefySession? session;
}
