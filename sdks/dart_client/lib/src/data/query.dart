/// A PostgREST query builder — the data layer for Supabase and Neon.
///
/// Chain filters and modifiers, then `await` the builder (or call [execute]):
///
/// ```dart
/// final rows = await client
///     .from('todos')
///     .select('id, title, done')
///     .eq('user_id', userId)
///     .order('created_at', ascending: false)
///     .limit(20);
/// ```
///
/// Writes:
/// ```dart
/// await client.from('todos').insert({'title': 'Ship SDK'});
/// await client.from('todos').update({'done': true}).eq('id', id);
/// await client.from('todos').delete().eq('id', id);
/// ```
library;

import 'dart:async';

import '../http.dart';

/// Built by [ForgefyClient.from]. Each terminal operation (`select`, `insert`,
/// `update`, `delete`, `upsert`) sets the HTTP verb; filters and modifiers
/// append PostgREST query parameters.
///
/// Implements `Future` so it can be awaited directly. The request fires when
/// the future is first awaited/`.then`-ed, exactly once.
class ForgefyQuery implements Future<Object?> {
  ForgefyQuery(this._http, this._restUrl, this._table);

  final ForgefyHttp _http;
  final String _restUrl;
  final String _table;

  String _method = 'GET';
  Object? _body;
  final List<String> _params = [];
  final Map<String, String> _headers = {};
  bool _single = false;

  Future<Object?>? _pending;

  // ── Terminal verbs ─────────────────────────────────────────────────────────

  /// Read rows. [columns] is a PostgREST select list, e.g. `'id, title'` or
  /// `'*, author(name)'` for an embedded resource.
  ForgefyQuery select([String columns = '*']) {
    _method = 'GET';
    _params.add('select=${Uri.encodeQueryComponent(columns)}');
    return this;
  }

  /// Insert one row (a `Map`) or many (a `List<Map>`). Returns the inserted
  /// rows.
  ForgefyQuery insert(Object values) {
    _method = 'POST';
    _body = values;
    _headers['Prefer'] = 'return=representation';
    return this;
  }

  /// Insert-or-update on conflict. [onConflict] names the unique column(s) that
  /// define a duplicate; omit to use the table's primary key.
  ForgefyQuery upsert(Object values, {String? onConflict}) {
    _method = 'POST';
    _body = values;
    _headers['Prefer'] = 'return=representation,resolution=merge-duplicates';
    if (onConflict != null) {
      _params.add('on_conflict=${Uri.encodeQueryComponent(onConflict)}');
    }
    return this;
  }

  /// Patch rows matching the filters. Combine with `.eq(...)` etc. — PostgREST
  /// refuses an unfiltered update.
  ForgefyQuery update(Map<String, dynamic> values) {
    _method = 'PATCH';
    _body = values;
    _headers['Prefer'] = 'return=representation';
    return this;
  }

  /// Delete rows matching the filters. Combine with a filter — PostgREST
  /// refuses an unfiltered delete.
  ForgefyQuery delete() {
    _method = 'DELETE';
    _headers['Prefer'] = 'return=representation';
    return this;
  }

  // ── Filters ────────────────────────────────────────────────────────────────

  ForgefyQuery eq(String column, Object value) => _filter(column, 'eq', value);
  ForgefyQuery neq(String column, Object value) => _filter(column, 'neq', value);
  ForgefyQuery gt(String column, Object value) => _filter(column, 'gt', value);
  ForgefyQuery gte(String column, Object value) => _filter(column, 'gte', value);
  ForgefyQuery lt(String column, Object value) => _filter(column, 'lt', value);
  ForgefyQuery lte(String column, Object value) => _filter(column, 'lte', value);

  /// SQL `LIKE` (case-sensitive). Use `%` as the wildcard.
  ForgefyQuery like(String column, String pattern) =>
      _filter(column, 'like', pattern);

  /// SQL `ILIKE` (case-insensitive).
  ForgefyQuery ilike(String column, String pattern) =>
      _filter(column, 'ilike', pattern);

  /// `column IS value`, for `null` / `true` / `false`.
  ForgefyQuery isFilter(String column, Object? value) =>
      _filter(column, 'is', value ?? 'null');

  /// `column IN (values)`.
  ForgefyQuery inFilter(String column, List<Object> values) =>
      _filter(column, 'in', '(${values.join(',')})');

  // ── Modifiers ──────────────────────────────────────────────────────────────

  ForgefyQuery order(String column, {bool ascending = true}) {
    _params.add(
      'order=${Uri.encodeQueryComponent('$column.${ascending ? 'asc' : 'desc'}')}',
    );
    return this;
  }

  ForgefyQuery limit(int count) {
    _params.add('limit=$count');
    return this;
  }

  /// Zero-based inclusive row window (like SQL `OFFSET`/`LIMIT`).
  ForgefyQuery range(int from, int to) {
    _params.add('offset=$from');
    _params.add('limit=${to - from + 1}');
    return this;
  }

  /// Return a single object instead of a list, and 406 if not exactly one row.
  ForgefyQuery single() {
    _single = true;
    _headers['Accept'] = 'application/vnd.pgrst.object+json';
    return this;
  }

  // ── Execution ──────────────────────────────────────────────────────────────

  /// Fire the request (idempotent — repeated awaits reuse the same future).
  Future<Object?> execute() => _pending ??= _run();

  Future<Object?> _run() async {
    final query = _params.isEmpty ? '' : '?${_params.join('&')}';
    final res = await _http.send(
      _method,
      '$_restUrl/$_table$query',
      body: _body,
      headers: _headers,
      // GET is safe to retry; writes carry no idempotency key, so are not.
      retryOn5xx: _method == 'GET',
    );
    // With `Prefer: return=representation` PostgREST replies with a list even
    // for single-row writes; `.single()` unwraps it.
    if (_single && res.data is List<dynamic>) {
      final list = res.data as List<dynamic>;
      return list.isEmpty ? null : list.first;
    }
    return res.data;
  }

  ForgefyQuery _filter(String column, String op, Object value) {
    _params.add(
      '${Uri.encodeQueryComponent(column)}=${Uri.encodeQueryComponent('$op.$value')}',
    );
    return this;
  }

  // ── Future delegation ──────────────────────────────────────────────────────

  @override
  Future<R> then<R>(FutureOr<R> Function(Object? value) onValue,
          {Function? onError}) =>
      execute().then(onValue, onError: onError);

  @override
  Future<Object?> catchError(Function onError,
          {bool Function(Object error)? test}) =>
      execute().catchError(onError, test: test);

  @override
  Future<Object?> whenComplete(FutureOr<void> Function() action) =>
      execute().whenComplete(action);

  @override
  Stream<Object?> asStream() => execute().asStream();

  @override
  Future<Object?> timeout(Duration timeLimit,
          {FutureOr<Object?> Function()? onTimeout}) =>
      execute().timeout(timeLimit, onTimeout: onTimeout);
}
