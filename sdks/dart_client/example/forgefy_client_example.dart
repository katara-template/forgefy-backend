// A minimal end-to-end walk-through of the Forgefy app-runtime SDK.
//
// Run against a real Supabase project:
//   dart run example/forgefy_client_example.dart
import 'package:forgefy_client/forgefy_client.dart';

Future<void> main() async {
  final client = ForgefyClient(
    ForgefyConfig(
      url: 'https://your-project.supabase.co',
      anonKey: 'your-anon-key',
    ),
    // A Flutter app would pass a persistent SessionStore here (e.g. one backed
    // by shared_preferences) so users stay signed in across restarts.
  );

  // React to auth changes anywhere in the app.
  client.auth.onAuthStateChange.listen((change) {
    print('auth: ${change.event.name} (user=${change.session?.user?.id})');
  });

  try {
    // Restore a persisted session on startup (no-op with the in-memory store).
    await client.auth.restoreSession();

    // Sign in — every query after this automatically carries the user's token.
    await client.auth.signInWithPassword(
      email: 'demo@example.com',
      password: 'super-secret',
    );

    // Create a row.
    final created = await client
        .from('todos')
        .insert({'title': 'Try the Forgefy SDK', 'done': false}).single();
    print('created: $created');

    // Read the current user's open todos, newest first.
    final open = await client
        .from('todos')
        .select('id, title, created_at')
        .eq('done', false)
        .order('created_at', ascending: false)
        .limit(20);
    print('open todos: $open');

    await client.auth.signOut();
  } on AuthException catch (e) {
    print('auth failed (${e.status}): ${e.message}');
  } on ForgefyException catch (e) {
    print('request failed (${e.status}): ${e.message}');
  } finally {
    await client.close();
  }
}
