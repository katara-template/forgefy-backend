/// Official Dart & Flutter SDK for the Forgefy Developer API — transcripts in,
/// structured product requirements (features, open questions, conflicts, action
/// items) out.
///
/// ```dart
/// import 'package:forgefy/forgefy.dart';
///
/// final forgefy = Forgefy(
///   apiKey: 'fgy_live_...',
///   baseUrl: 'https://your-forgefy-host',
/// );
///
/// final result = await forgefy.extract(
///   transcript: 'We need Google login before launch. Sarah owns billing.',
/// );
/// for (final f in result.features) {
///   print('[${f.priority}] ${f.title}');
/// }
/// forgefy.close();
/// ```
library;

export 'src/client.dart' show Forgefy, Jobs;
export 'src/errors.dart';
export 'src/models.dart';
export 'src/webhooks.dart' show verifySignature;
