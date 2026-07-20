/// Typed exceptions surfaced by the Forgefy app-runtime SDK.
///
/// Backends disagree on their error envelopes — GoTrue uses
/// `{error, error_description}` (and, on newer versions, `{code, msg}`),
/// PostgREST uses `{message, details, hint, code}`, and some gateways return
/// RFC 7807 `{type, title, detail}`. [errorFromResponse] normalises all three
/// into one hierarchy so app code can `catch` by meaning, not by provider.
///
/// Dart convention names runtime failures `Exception` (not `Error`, which is
/// reserved for programming mistakes).
library;

/// Base class for every failure surfaced by the SDK.
class ForgefyException implements Exception {
  ForgefyException(
    this.message, {
    this.status = 0,
    this.code,
    this.details,
  });

  /// Human-readable message, best-effort extracted from the backend's body.
  final String message;

  /// HTTP status code, or 0 when no response was received.
  final int status;

  /// Backend-specific error code, when present (e.g. PostgREST `23505`,
  /// GoTrue `invalid_credentials`).
  final String? code;

  /// Extra context (PostgREST `details`/`hint`), when present.
  final String? details;

  @override
  String toString() => '$runtimeType($status): $message';
}

/// 400 / 422 — the request was malformed or failed validation.
class ValidationException extends ForgefyException {
  ValidationException(super.message, {super.status, super.code, super.details});
}

/// 401 / 403 — not signed in, token expired, or blocked by row-level security.
class AuthException extends ForgefyException {
  AuthException(super.message, {super.status, super.code, super.details});
}

/// 404 — the resource doesn't exist or is hidden by access rules.
class NotFoundException extends ForgefyException {
  NotFoundException(super.message, {super.status, super.code, super.details});
}

/// 409 — a conflict, most often a unique-constraint violation on insert/upsert.
class ConflictException extends ForgefyException {
  ConflictException(super.message, {super.status, super.code, super.details});
}

/// 429 — rate limited. Honour `Retry-After` (the SDK already retries it once).
class RateLimitException extends ForgefyException {
  RateLimitException(super.message, {super.status, super.code, super.details});
}

/// 5xx — the backend faulted.
class ServerException extends ForgefyException {
  ServerException(super.message, {super.status, super.code, super.details});
}

/// The request never got an HTTP response (DNS, connection refused, timeout).
class ConnectionException extends ForgefyException {
  ConnectionException(super.message);
}

/// Map an HTTP status + decoded body to the matching exception subclass.
ForgefyException errorFromResponse(int status, Object? body) {
  final map = body is Map<String, dynamic> ? body : const <String, dynamic>{};
  final message = _messageFrom(map, status);
  final code = _stringOf(map['code']) ?? _stringOf(map['error']);
  final details = _stringOf(map['details']) ?? _stringOf(map['hint']);

  switch (status) {
    case 400:
    case 422:
      return ValidationException(message,
          status: status, code: code, details: details);
    case 401:
    case 403:
      return AuthException(message,
          status: status, code: code, details: details);
    case 404:
      return NotFoundException(message,
          status: status, code: code, details: details);
    case 409:
      return ConflictException(message,
          status: status, code: code, details: details);
    case 429:
      return RateLimitException(message,
          status: status, code: code, details: details);
    default:
      if (status >= 500) {
        return ServerException(message,
            status: status, code: code, details: details);
      }
      return ForgefyException(message,
          status: status, code: code, details: details);
  }
}

/// Pull the most descriptive message out of whichever envelope the backend used.
String _messageFrom(Map<String, dynamic> body, int status) {
  final candidate = _stringOf(body['error_description']) ?? // GoTrue (older)
      _stringOf(body['msg']) ?? // GoTrue (newer)
      _stringOf(body['message']) ?? // PostgREST
      _stringOf(body['detail']) ?? // RFC 7807
      _stringOf(body['title']) ?? // RFC 7807
      _stringOf(body['error']); // last resort (may be a code, but better than nothing)
  if (candidate != null && candidate.isNotEmpty) return candidate;
  return 'Request failed with status $status';
}

String? _stringOf(Object? value) => value is String ? value : null;
