/// Forgefy app-runtime SDK for Dart & Flutter.
///
/// The auth, data, and networking layer that Forgefy-generated apps depend on,
/// so the build agent wires in a tested primitive instead of regenerating a
/// `core/` layer on every build.
library;

export 'src/auth/auth.dart';
export 'src/auth/models.dart';
export 'src/auth/session_store.dart';
export 'src/client.dart';
export 'src/config.dart';
export 'src/data/query.dart';
export 'src/errors.dart';
export 'src/http.dart' show ForgefyResponse;
