import 'dart:convert';

import 'package:forgefy/forgefy.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

const _tinyDelay = Duration(milliseconds: 1);

Forgefy _client(MockClient mock) => Forgefy(
      apiKey: 'fgy_test_123',
      baseUrl: 'https://api.example.test',
      httpClient: mock,
      retryDelay: _tinyDelay,
    );

void main() {
  group('extract', () {
    test('sends auth + body and parses the result', () async {
      late http.Request captured;
      final mock = MockClient((req) async {
        captured = req;
        return http.Response(
          jsonEncode({
            'id': 'abc',
            'model_tier': 'standard',
            'features': [
              {'title': 'Google login', 'description': 'OAuth', 'priority': 'high'}
            ],
            'questions': [],
            'conflicts': [],
            'action_items': [],
            'usage': {'input_tokens': 10, 'output_tokens': 20},
          }),
          200,
          headers: {'content-type': 'application/json'},
        );
      });

      final forgefy = _client(mock);
      final result = await forgefy.extract(
        transcript: 'We need Google login',
        extractors: [Extractor.features, Extractor.actionItems],
      );

      expect(captured.method, 'POST');
      expect(captured.url.path, '/api/v1/extract');
      expect(captured.headers['authorization'], 'Bearer fgy_test_123');
      final sent = jsonDecode(captured.body) as Map<String, dynamic>;
      expect(sent['extractors'], ['features', 'action_items']);

      expect(result.id, 'abc');
      expect(result.features.single.title, 'Google login');
      expect(result.usage.inputTokens, 10);
    });

    test('retries on 429 then succeeds', () async {
      var calls = 0;
      final mock = MockClient((req) async {
        calls++;
        if (calls == 1) return http.Response('{"detail":"slow down"}', 429);
        return http.Response(
          jsonEncode({
            'id': 'x',
            'model_tier': 'standard',
            'features': [],
            'questions': [],
            'conflicts': [],
            'action_items': [],
            'usage': {'input_tokens': 0, 'output_tokens': 0},
          }),
          200,
        );
      });

      final result = await _client(mock).extract(transcript: 'hi there friend');
      expect(calls, 2);
      expect(result.id, 'x');
    });

    test('maps 401 to AuthenticationException', () async {
      final mock = MockClient(
        (req) async => http.Response('{"detail":"bad key"}', 401),
      );
      expect(
        () => _client(mock).extract(transcript: 'hello world here'),
        throwsA(isA<AuthenticationException>()
            .having((e) => e.status, 'status', 401)
            .having((e) => e.detail, 'detail', 'bad key')),
      );
    });
  });

  group('jobs', () {
    test('create sends an Idempotency-Key', () async {
      late http.Request captured;
      final mock = MockClient((req) async {
        captured = req;
        return http.Response(
          jsonEncode(
              {'job_id': 'job-1', 'status': 'queued', 'webhook_secret': 'sek'}),
          202,
        );
      });

      final job = await _client(mock).jobs.create(transcript: 'long transcript here');
      expect(captured.headers['idempotency-key'], isNotEmpty);
      expect(job.jobId, 'job-1');
      expect(job.status, JobStatus.queued);
      expect(job.webhookSecret, 'sek');
    });

    test('waitFor polls until done', () async {
      var calls = 0;
      final mock = MockClient((req) async {
        calls++;
        final status = calls < 2 ? 'processing' : 'done';
        return http.Response(
          jsonEncode({
            'job_id': 'job-2',
            'status': status,
            'model_tier': 'standard',
            'created_at': '2026-01-01T00:00:00Z',
            'result': status == 'done' ? {'features': []} : null,
            'error': null,
          }),
          200,
        );
      });

      final job = await _client(mock)
          .jobs
          .waitFor('job-2', pollInterval: _tinyDelay);
      expect(job.status, JobStatus.done);
      expect(calls, 2);
    });

    test('waitFor throws JobFailedException on failure', () async {
      final mock = MockClient(
        (req) async => http.Response(
          jsonEncode({
            'job_id': 'job-3',
            'status': 'failed',
            'model_tier': 'standard',
            'created_at': '2026-01-01T00:00:00Z',
            'result': null,
            'error': 'all extractors errored',
          }),
          200,
        ),
      );
      expect(
        () => _client(mock).jobs.waitFor('job-3', pollInterval: _tinyDelay),
        throwsA(isA<JobFailedException>()
            .having((e) => e.jobId, 'jobId', 'job-3')),
      );
    });
  });

  group('verifySignature', () {
    // HMAC-SHA256 of 'hello' keyed by 'secret', hex.
    const body = 'hello';
    const secret = 'secret';
    const validHex =
        '88aab3ede8d3adf94d26ab90d3bafd4a2083070c3bcce9c014ee04a443847c0b';

    test('accepts a valid sha256= signature', () {
      expect(verifySignature(body, 'sha256=$validHex', secret), isTrue);
    });

    test('accepts a bare hex signature', () {
      expect(verifySignature(body, validHex, secret), isTrue);
    });

    test('rejects a wrong signature, secret, or missing header', () {
      expect(verifySignature(body, 'sha256=${'0' * 64}', secret), isFalse);
      expect(verifySignature(body, 'sha256=$validHex', 'wrong'), isFalse);
      expect(verifySignature(body, null, secret), isFalse);
      expect(verifySignature(body, 'sha256=notarealhash', secret), isFalse);
    });
  });
}
