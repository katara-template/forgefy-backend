/// Typed exceptions mapped from the API's RFC 7807 problem+json responses.
///
/// Dart convention names runtime failures `Exception` rather than `Error`
/// (`Error` is reserved for programming mistakes), so these mirror the
/// TypeScript/Python `*Error` classes one-to-one under Dart naming.
library;

/// Base class for every failure surfaced by the SDK.
class ForgefyException implements Exception {
  ForgefyException(
    this.message, {
    this.status = 0,
    this.detail,
    this.problemType,
  });

  /// Human-readable message (the problem `detail`, then `title`, then a default).
  final String message;

  /// HTTP status code, or 0 when no response was received.
  final int status;

  /// The problem+json `detail`, when present.
  final String? detail;

  /// The problem+json `type` URI, when present.
  final String? problemType;

  @override
  String toString() => '$runtimeType($status): $message';
}

/// 401 — missing, malformed, or revoked API key.
class AuthenticationException extends ForgefyException {
  AuthenticationException(super.message,
      {super.status, super.detail, super.problemType});
}

/// 402 — monthly token budget exhausted (free tier). `detail` has the reset date.
class QuotaExceededException extends ForgefyException {
  QuotaExceededException(super.message,
      {super.status, super.detail, super.problemType});
}

/// 404 — the resource doesn't exist or belongs to another account.
class NotFoundException extends ForgefyException {
  NotFoundException(super.message,
      {super.status, super.detail, super.problemType});
}

/// 422 — invalid request (empty/oversized transcript, bad extractor name, …).
class ValidationException extends ForgefyException {
  ValidationException(super.message,
      {super.status, super.detail, super.problemType});
}

/// 429 — over 60 req/min for this key, or the server is at sync capacity.
class RateLimitException extends ForgefyException {
  RateLimitException(super.message,
      {super.status, super.detail, super.problemType});
}

/// 5xx — extraction failed on every model, queue unreachable, or other fault.
class ServerException extends ForgefyException {
  ServerException(super.message,
      {super.status, super.detail, super.problemType});
}

/// The request never got an HTTP response (DNS, refused, timeout).
class ApiConnectionException extends ForgefyException {
  // status defaults to 0 in the base constructor — no response was received.
  ApiConnectionException(super.message);
}

/// `jobs.waitFor`: the job finished with status "failed".
class JobFailedException extends ForgefyException {
  JobFailedException(this.jobId, String? detail)
      : super(
          'Extract job $jobId failed${detail != null && detail.isNotEmpty ? ': $detail' : ''}',
          detail: detail,
        );

  final String jobId;
}

/// `jobs.waitFor`: the deadline passed before the job finished.
class JobTimeoutException extends ForgefyException {
  JobTimeoutException(this.jobId, Duration timeout)
      : super(
          'Timed out after ${timeout.inMilliseconds}ms waiting for extract job $jobId',
        );

  final String jobId;
}

/// Map an HTTP status + problem+json body to the matching exception subclass.
ForgefyException errorFromProblem(int status, Map<String, dynamic> problem) {
  final message = (problem['detail'] as String?) ??
      (problem['title'] as String?) ??
      'Forgefy API error $status';
  final detail = problem['detail'] as String?;
  final type = problem['type'] as String?;

  switch (status) {
    case 401:
      return AuthenticationException(message,
          status: status, detail: detail, problemType: type);
    case 402:
      return QuotaExceededException(message,
          status: status, detail: detail, problemType: type);
    case 404:
      return NotFoundException(message,
          status: status, detail: detail, problemType: type);
    case 422:
      return ValidationException(message,
          status: status, detail: detail, problemType: type);
    case 429:
      return RateLimitException(message,
          status: status, detail: detail, problemType: type);
    default:
      if (status >= 500) {
        return ServerException(message,
            status: status, detail: detail, problemType: type);
      }
      return ForgefyException(message,
          status: status, detail: detail, problemType: type);
  }
}
