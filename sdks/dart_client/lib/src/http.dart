/// Low-level HTTP transport: header injection, retries, and error mapping.
///
/// The retry/backoff/idempotency logic is ported from the Forgefy Developer
/// SDK's proven client — it is the one piece every generated app used to
/// hand-roll (as `core/network/api_client.dart`) and get subtly wrong.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:http/http.dart' as http;

import 'config.dart';
import 'errors.dart';

const _userAgent = 'forgefy-client-dart/0.1.0';

/// A decoded HTTP response. [headers] is retained because PostgREST returns
/// pagination info in `Content-Range`.
class ForgefyResponse {
  ForgefyResponse(this.status, this.data, this.headers);

  final int status;

  /// JSON-decoded body — a `Map`, a `List`, or `null` for empty responses.
  final Object? data;

  final Map<String, String> headers;
}

/// Returns the current user access token, or `null` before sign-in.
typedef TokenProvider = String? Function();

/// Shared transport used by both the auth and data layers so a single signed-in
/// session's token flows to every request.
class ForgefyHttp {
  ForgefyHttp(this._config, {http.Client? httpClient})
      : _http = httpClient ?? http.Client(),
        _ownsClient = httpClient == null;

  final ForgefyConfig _config;
  final http.Client _http;
  final bool _ownsClient;
  final Random _rng = Random.secure();

  /// Set by [ForgefyClient] to point at the live session's access token.
  TokenProvider? tokenProvider;

  /// Perform one request against a fully-built [url], with retries.
  ///
  /// 429 is always retried (honouring `Retry-After`). 5xx is retried only when
  /// [retryOn5xx] is set — safe for reads and for writes carrying an
  /// idempotency key, never for a plain non-idempotent POST.
  Future<ForgefyResponse> send(
    String method,
    String url, {
    Object? body,
    Map<String, String>? headers,
    bool retryOn5xx = false,
  }) async {
    final uri = Uri.parse(url);
    final token = tokenProvider?.call() ?? _config.anonKey;
    final requestHeaders = <String, String>{
      if (_config.sendsApiKeyHeader) 'apikey': _config.anonKey,
      'Authorization': 'Bearer $token',
      'Content-Type': 'application/json',
      'User-Agent': _userAgent,
      ...?headers,
    };
    final encodedBody = body == null ? null : jsonEncode(body);

    var attempt = 0;
    while (true) {
      http.Response response;
      try {
        response = await _dispatch(method, uri, requestHeaders, encodedBody);
      } catch (err) {
        if (attempt < _config.maxRetries) {
          await Future<void>.delayed(_backoff(attempt));
          attempt++;
          continue;
        }
        throw ConnectionException('Could not reach $uri: $err');
      }

      final status = response.statusCode;
      if (status >= 200 && status < 300) {
        return ForgefyResponse(status, _decode(response.body), response.headers);
      }

      final retryable = status == 429 || (retryOn5xx && status >= 500);
      if (retryable && attempt < _config.maxRetries) {
        final retryAfter = _parseRetryAfter(response.headers['retry-after']);
        await Future<void>.delayed(retryAfter ?? _backoff(attempt));
        attempt++;
        continue;
      }

      throw errorFromResponse(status, _decode(response.body));
    }
  }

  /// Release the underlying client. No-op when a client was injected — the
  /// caller owns its lifecycle then.
  void close() {
    if (_ownsClient) _http.close();
  }

  Future<http.Response> _dispatch(
    String method,
    Uri url,
    Map<String, String> headers,
    String? body,
  ) {
    final request = http.Request(method, url);
    request.headers.addAll(headers);
    if (body != null) request.bodyBytes = utf8.encode(body);
    return _http
        .send(request)
        .timeout(_config.timeout)
        .then(http.Response.fromStream);
  }

  Object? _decode(String body) {
    if (body.isEmpty) return null;
    try {
      return jsonDecode(body);
    } catch (_) {
      // Non-JSON body (e.g. a proxy's HTML error page) — surface it as text.
      return {'message': body};
    }
  }

  Duration _backoff(int attempt) {
    // 1x, 2x, 4x… the base delay, with a little jitter.
    final base = _config.retryDelay.inMilliseconds * pow(2, attempt);
    final jittered = base * (0.8 + _rng.nextDouble() * 0.4);
    return Duration(milliseconds: jittered.round());
  }
}

Duration? _parseRetryAfter(String? value) {
  if (value == null) return null;
  final seconds = double.tryParse(value);
  if (seconds == null || seconds <= 0) return null;
  return Duration(milliseconds: (seconds * 1000).round());
}
