import 'dart:convert';

import 'package:forgefy_client/forgefy_client.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:test/test.dart';

/// Builds a client whose transport records every request and replies with
/// [handler], so tests can assert on the exact URL/verb/headers/body sent.
({ForgefyClient client, List<http.Request> sent}) _harness(
  Future<http.Response> Function(http.Request req) handler,
) {
  final sent = <http.Request>[];
  final mock = MockClient((req) async {
    sent.add(req);
    return handler(req);
  });
  final client = ForgefyClient(
    ForgefyConfig(url: 'https://proj.supabase.co', anonKey: 'anon-key'),
    httpClient: mock,
  );
  return (client: client, sent: sent);
}

http.Response _json(Object body, {int status = 200}) =>
    http.Response(jsonEncode(body), status,
        headers: {'content-type': 'application/json'});

void main() {
  group('data query building', () {
    test('select with filters, order and limit → correct URL and verb', () async {
      final h = _harness((_) async => _json([
            {'id': 1, 'title': 'a'}
          ]));

      final rows = await h.client
          .from('todos')
          .select('id, title')
          .eq('done', false)
          .order('created_at', ascending: false)
          .limit(20);

      expect(rows, isA<List<dynamic>>());
      final req = h.sent.single;
      expect(req.method, 'GET');
      expect(req.url.path, '/rest/v1/todos');
      final q = req.url.query;
      expect(q, contains('select=id%2C%20title'));
      expect(q, contains('done=eq.false'));
      expect(q, contains('order=created_at.desc'));
      expect(q, contains('limit=20'));
    });

    test('insert → POST with return=representation and body', () async {
      final h = _harness((_) async => _json([
            {'id': 7, 'title': 'Ship SDK'}
          ]));

      await h.client.from('todos').insert({'title': 'Ship SDK'});

      final req = h.sent.single;
      expect(req.method, 'POST');
      expect(req.headers['Prefer'], contains('return=representation'));
      expect(jsonDecode(req.body), {'title': 'Ship SDK'});
    });

    test('update with filter → PATCH', () async {
      final h = _harness((_) async => _json([]));
      await h.client.from('todos').update({'done': true}).eq('id', 7);

      final req = h.sent.single;
      expect(req.method, 'PATCH');
      expect(req.url.query, contains('id=eq.7'));
    });

    test('single() unwraps the representation list to one object', () async {
      final h = _harness((_) async => _json([
            {'id': 7, 'title': 'a'}
          ]));

      final row = await h.client.from('todos').insert({'title': 'a'}).single();
      expect(row, {'id': 7, 'title': 'a'});
    });

    test('anon key is sent before sign-in', () async {
      final h = _harness((_) async => _json([]));
      await h.client.from('todos').select();

      final req = h.sent.single;
      expect(req.headers['apikey'], 'anon-key');
      expect(req.headers['authorization'], 'Bearer anon-key');
    });
  });

  group('auth', () {
    test('signIn parses the session and later queries carry the user token',
        () async {
      final h = _harness((req) async {
        if (req.url.path.endsWith('/token')) {
          return _json({
            'access_token': 'user-jwt',
            'refresh_token': 'refresh-1',
            'expires_in': 3600,
            'user': {'id': 'u1', 'email': 'a@b.c'},
          });
        }
        return _json([]);
      });

      final session = await h.client.auth
          .signInWithPassword(email: 'a@b.c', password: 'pw');
      expect(session.isAuthenticated, isTrue);
      expect(h.client.auth.currentUser?.id, 'u1');

      await h.client.from('todos').select();
      final query = h.sent.last;
      expect(query.headers['authorization'], 'Bearer user-jwt');
      expect(query.headers['apikey'], 'anon-key');
    });

    test('onAuthStateChange emits signedIn then signedOut', () async {
      final h = _harness((req) async {
        if (req.url.path.endsWith('/token')) {
          return _json({
            'access_token': 'user-jwt',
            'refresh_token': 'r',
            'expires_in': 3600,
            'user': {'id': 'u1'},
          });
        }
        return _json({}); // logout
      });

      final events = <AuthChangeEvent>[];
      h.client.auth.onAuthStateChange.listen((c) => events.add(c.event));

      await h.client.auth.signInWithPassword(email: 'a@b.c', password: 'pw');
      await h.client.auth.signOut();
      await Future<void>.delayed(Duration.zero); // let the stream drain

      expect(events, [AuthChangeEvent.signedIn, AuthChangeEvent.signedOut]);
      expect(h.client.auth.currentSession, isNull);
    });

    test('restoreSession refreshes an expired stored session', () async {
      final store = InMemorySessionStore();
      final expired = ForgefySession(
        accessToken: 'old',
        refreshToken: 'refresh-1',
        expiresAt: 0, // long past
        user: ForgefyUser(id: 'u1'),
      );
      await store.write(jsonEncode(expired.toJson()));

      var refreshCalls = 0;
      final mock = MockClient((req) async {
        if (req.url.query.contains('grant_type=refresh_token')) {
          refreshCalls++;
          return _json({
            'access_token': 'fresh-jwt',
            'refresh_token': 'refresh-2',
            'expires_in': 3600,
            'user': {'id': 'u1'},
          });
        }
        return _json({});
      });
      final client = ForgefyClient(
        ForgefyConfig(url: 'https://proj.supabase.co', anonKey: 'anon-key'),
        sessionStore: store,
        httpClient: mock,
      );

      final restored = await client.auth.restoreSession();
      expect(refreshCalls, 1);
      expect(restored?.accessToken, 'fresh-jwt');
    });
  });

  group('errors', () {
    test('401 → AuthException', () async {
      final h = _harness(
          (_) async => _json({'msg': 'invalid token'}, status: 401));
      await expectLater(
        h.client.from('todos').select(),
        throwsA(isA<AuthException>()
            .having((e) => e.status, 'status', 401)
            .having((e) => e.message, 'message', 'invalid token')),
      );
    });

    test('409 unique violation → ConflictException with PostgREST fields',
        () async {
      final h = _harness((_) async => _json({
            'message': 'duplicate key value violates unique constraint',
            'code': '23505',
            'details': 'Key (email)=(a@b.c) already exists.',
          }, status: 409));

      await expectLater(
        h.client.from('users').insert({'email': 'a@b.c'}),
        throwsA(isA<ConflictException>()
            .having((e) => e.code, 'code', '23505')
            .having((e) => e.details, 'details', contains('already exists'))),
      );
    });
  });

  group('session model', () {
    test('isExpired respects leeway and absolute expiry', () {
      final now = DateTime.now().millisecondsSinceEpoch ~/ 1000;
      final soon = ForgefySession(accessToken: 't', expiresAt: now + 10);
      final later = ForgefySession(accessToken: 't', expiresAt: now + 3600);
      expect(soon.isExpired(leeway: const Duration(seconds: 30)), isTrue);
      expect(later.isExpired(leeway: const Duration(seconds: 30)), isFalse);
    });

    test('a confirmation-pending signup body parses as unauthenticated', () {
      final s = ForgefySession.fromJson({'id': 'u1', 'email': 'a@b.c'});
      expect(s.isAuthenticated, isFalse);
      expect(s.user?.id, 'u1');
    });
  });
}
