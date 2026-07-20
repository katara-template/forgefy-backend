/// Runtime configuration for a Forgefy-generated app's backend.
library;

/// The backend a generated app was wired to at build time.
///
/// Both [supabase] and [neon] speak PostgREST for data, so they share the data
/// layer; only [supabase] currently provides auth. [firebase] is reserved — it
/// will slot in behind the same [ForgefyAuth]/data interfaces later.
enum ForgefyProvider { supabase, neon, firebase }

/// Immutable connection settings.
///
/// Only public-safe values live here — a Supabase URL + anon key, or a Neon
/// Data API URL + publishable key. Access is enforced by Row Level Security /
/// Postgres grants on the server, never by keeping the key secret. This is the
/// same contract the build agent already documents for generated apps.
class ForgefyConfig {
  ForgefyConfig({
    required String url,
    required this.anonKey,
    this.provider = ForgefyProvider.supabase,
    this.maxRetries = 2,
    this.timeout = const Duration(seconds: 30),
    this.retryDelay = const Duration(milliseconds: 500),
  }) : url = url.replaceAll(RegExp(r'/+$'), '') {
    if (this.url.isEmpty) throw ArgumentError('ForgefyConfig: url is required');
    if (anonKey.isEmpty) throw ArgumentError('ForgefyConfig: anonKey is required');
  }

  /// Project root, e.g. `https://abcd.supabase.co`, or the Neon Data API URL.
  /// Any trailing slash is stripped so path joins stay clean.
  final String url;

  /// Public API key (Supabase anon key / Neon publishable key). Sent as the
  /// `apikey` header on Supabase and as the fallback bearer token before sign-in.
  final String anonKey;

  final ForgefyProvider provider;
  final int maxRetries;
  final Duration timeout;
  final Duration retryDelay;

  /// GoTrue base for Supabase; auth is not supported on other providers yet.
  String get authUrl => '$url/auth/v1';

  /// PostgREST base. Supabase namespaces it under `/rest/v1`; a Neon Data API
  /// URL is already the PostgREST root.
  String get restUrl =>
      provider == ForgefyProvider.supabase ? '$url/rest/v1' : url;

  /// Whether the `apikey` header (a Supabase requirement) should be sent.
  bool get sendsApiKeyHeader => provider == ForgefyProvider.supabase;
}
