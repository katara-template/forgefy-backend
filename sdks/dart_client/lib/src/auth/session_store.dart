/// Pluggable persistence for the signed-in session.
///
/// The default ([InMemorySessionStore]) keeps the session for the life of the
/// process — fine for servers and tests. A Flutter app supplies a persistent
/// implementation (e.g. backed by `shared_preferences` or
/// `flutter_secure_storage`) so a user stays logged in across restarts. Keeping
/// this an interface means the SDK never forces a storage dependency on apps
/// that don't want one.
library;

/// Reads, writes, and clears the serialized session string.
abstract class SessionStore {
  Future<String?> read();
  Future<void> write(String value);
  Future<void> delete();
}

/// Non-persistent default. Lives only as long as the client.
class InMemorySessionStore implements SessionStore {
  String? _value;

  @override
  Future<String?> read() async => _value;

  @override
  Future<void> write(String value) async => _value = value;

  @override
  Future<void> delete() async => _value = null;
}
