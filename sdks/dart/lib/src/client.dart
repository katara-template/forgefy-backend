/// Forgefy API client.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:http/http.dart' as http;

import 'errors.dart';
import 'models.dart';

const _userAgent = 'forgefy-sdk-dart/0.1.0';
const _defaultBaseUrl = 'https://forgefy.onrender.com';

/// Client for the Forgefy Developer API.
///
/// Works on the Dart VM and Flutter (mobile, desktop, and web) — it depends
/// only on `package:http`, never `dart:io`.
///
/// ```dart
/// final forgefy = Forgefy(apiKey: 'fgy_live_...');
/// final result = await forgefy.extract(transcript: '...');
/// forgefy.close();
/// ```
class Forgefy {
  Forgefy({
    required String apiKey,
    String? baseUrl,
    int maxRetries = 2,
    Duration timeout = const Duration(seconds: 120),
    Duration retryDelay = const Duration(milliseconds: 500),
    http.Client? httpClient,
  })  : _apiKey = apiKey,
        _baseUrl = (baseUrl ?? _defaultBaseUrl).replaceAll(RegExp(r'/+$'), ''),
        _maxRetries = maxRetries,
        _timeout = timeout,
        _retryDelay = retryDelay,
        _ownsClient = httpClient == null,
        _http = httpClient ?? http.Client() {
    if (apiKey.isEmpty) {
      throw ArgumentError('Forgefy: apiKey is required');
    }
    jobs = Jobs._(this);
  }

  final String _apiKey;
  final String _baseUrl;
  final int _maxRetries;
  final Duration _timeout;
  final Duration _retryDelay;
  final bool _ownsClient;
  final http.Client _http;
  final Random _rng = Random.secure();

  /// Async extraction jobs (transcripts up to 200k chars).
  late final Jobs jobs;

  /// Synchronously extract features / questions / conflicts / action items from
  /// a transcript (≤ 50k chars). For longer transcripts use [jobs].create().
  ///
  /// Pass [extractors] to run only a subset (you're only metered for those);
  /// omit it to run all four.
  Future<ExtractResult> extract({
    required String transcript,
    List<Extractor>? extractors,
    ModelTier? modelTier,
  }) async {
    final json = await _request(
      'POST',
      '/api/v1/extract',
      body: _extractBody(transcript, extractors, modelTier),
    ) as Map<String, dynamic>;
    return ExtractResult.fromJson(json);
  }

  /// The key owner's tier, monthly token budget, consumption, and reset date.
  Future<UsageResponse> usage() async {
    final json = await _request('GET', '/api/v1/usage', retryOn5xx: true)
        as Map<String, dynamic>;
    return UsageResponse.fromJson(json);
  }

  /// Close the underlying HTTP client. No-op when a client was injected (the
  /// caller owns its lifecycle then).
  void close() {
    if (_ownsClient) _http.close();
  }

  // ── Internals ──────────────────────────────────────────────────────────────

  /// One API call with retries.
  ///
  /// 429 is always retried (with `Retry-After` honored). 5xx is retried only
  /// when [retryOn5xx] is set — GETs and idempotent job creation; never the
  /// sync `/extract` POST, where a retry would bill tokens twice for work that
  /// may have partially run.
  Future<dynamic> _request(
    String method,
    String path, {
    Object? body,
    Map<String, String>? headers,
    bool retryOn5xx = false,
  }) async {
    final url = Uri.parse('$_baseUrl$path');
    final requestHeaders = <String, String>{
      'Authorization': 'Bearer $_apiKey',
      'Content-Type': 'application/json',
      'User-Agent': _userAgent,
      ...?headers,
    };
    final encodedBody = body == null ? null : jsonEncode(body);

    var attempt = 0;
    while (true) {
      http.Response response;
      try {
        response = await _send(method, url, requestHeaders, encodedBody);
      } catch (err) {
        if (attempt < _maxRetries) {
          await Future<void>.delayed(_backoff(attempt));
          attempt++;
          continue;
        }
        throw ApiConnectionException(
          'Could not reach the Forgefy API at $_baseUrl: $err',
        );
      }

      final status = response.statusCode;
      if (status >= 200 && status < 300) {
        if (status == 204 || response.body.isEmpty) return null;
        return jsonDecode(response.body);
      }

      final retryable = status == 429 || (retryOn5xx && status >= 500);
      if (retryable && attempt < _maxRetries) {
        final retryAfter = _parseRetryAfter(response.headers['retry-after']);
        await Future<void>.delayed(retryAfter ?? _backoff(attempt));
        attempt++;
        continue;
      }

      throw errorFromProblem(status, _problemBody(response.body));
    }
  }

  Future<http.Response> _send(
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
        .timeout(_timeout)
        .then(http.Response.fromStream);
  }

  Duration _backoff(int attempt) {
    // 1x, 2x, 4x… the base delay, with a little jitter.
    final base = _retryDelay.inMilliseconds * pow(2, attempt);
    final jittered = base * (0.8 + _rng.nextDouble() * 0.4);
    return Duration(milliseconds: jittered.round());
  }

  /// A UUID v4, used for the auto-generated `Idempotency-Key`.
  String _uuidV4() {
    final b = List<int>.generate(16, (_) => _rng.nextInt(256));
    b[6] = (b[6] & 0x0f) | 0x40; // version 4
    b[8] = (b[8] & 0x3f) | 0x80; // variant 10
    final hex = b.map((x) => x.toRadixString(16).padLeft(2, '0')).join();
    return '${hex.substring(0, 8)}-${hex.substring(8, 12)}-'
        '${hex.substring(12, 16)}-${hex.substring(16, 20)}-${hex.substring(20)}';
  }
}

/// Async extraction jobs (transcripts up to 200k chars). Accessed via
/// [Forgefy.jobs].
class Jobs {
  Jobs._(this._client);

  final Forgefy _client;

  /// Queue an async extraction job (transcripts up to 200k chars).
  ///
  /// An `Idempotency-Key` is generated automatically so network-level retries
  /// can never run the same job twice; pass [idempotencyKey] to dedupe across
  /// processes (e.g. a stable import-batch id). When [webhookUrl] is given, the
  /// result is POSTed there on completion — verify it with [verifySignature]
  /// using the returned [JobCreated.webhookSecret].
  Future<JobCreated> create({
    required String transcript,
    List<Extractor>? extractors,
    ModelTier? modelTier,
    String? webhookUrl,
    String? idempotencyKey,
  }) async {
    final body = _extractBody(transcript, extractors, modelTier);
    if (webhookUrl != null) body['webhook_url'] = webhookUrl;

    final json = await _client._request(
      'POST',
      '/api/v1/extract/jobs',
      body: body,
      headers: {'Idempotency-Key': idempotencyKey ?? _client._uuidV4()},
      retryOn5xx: true, // safe: the idempotency key dedupes on the server
    ) as Map<String, dynamic>;
    return JobCreated.fromJson(json);
  }

  /// Current status of a job; `result` is set once status is [JobStatus.done].
  Future<JobStatusResponse> get(String jobId) async {
    final json = await _client._request(
      'GET',
      '/api/v1/extract/jobs/${Uri.encodeComponent(jobId)}',
      retryOn5xx: true,
    ) as Map<String, dynamic>;
    return JobStatusResponse.fromJson(json);
  }

  /// Poll a job until it finishes.
  ///
  /// Returns the "done" status response, throws [JobFailedException] on
  /// "failed" and [JobTimeoutException] when [timeout] passes first.
  Future<JobStatusResponse> waitFor(
    String jobId, {
    Duration pollInterval = const Duration(seconds: 3),
    Duration timeout = const Duration(minutes: 10),
  }) async {
    final deadline = DateTime.now().add(timeout);
    while (true) {
      final job = await get(jobId);
      if (job.status == JobStatus.done) return job;
      if (job.status == JobStatus.failed) {
        throw JobFailedException(jobId, job.error);
      }
      if (DateTime.now().add(pollInterval).isAfter(deadline)) {
        throw JobTimeoutException(jobId, timeout);
      }
      await Future<void>.delayed(pollInterval);
    }
  }
}

Map<String, dynamic> _extractBody(
  String transcript,
  List<Extractor>? extractors,
  ModelTier? modelTier,
) {
  final body = <String, dynamic>{'transcript': transcript};
  if (extractors != null) {
    body['extractors'] = extractors.map((e) => e.wire).toList();
  }
  if (modelTier != null) body['model_tier'] = modelTier.wire;
  return body;
}

Map<String, dynamic> _problemBody(String body) {
  if (body.isEmpty) return const {};
  try {
    final decoded = jsonDecode(body);
    return decoded is Map<String, dynamic> ? decoded : const {};
  } catch (_) {
    return const {};
  }
}

Duration? _parseRetryAfter(String? value) {
  if (value == null) return null;
  final seconds = double.tryParse(value);
  if (seconds == null || seconds <= 0) return null;
  return Duration(milliseconds: (seconds * 1000).round());
}
