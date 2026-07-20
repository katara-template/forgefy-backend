/// Webhook signature verification.
library;

import 'dart:convert';

import 'package:crypto/crypto.dart';

/// Verify a Forgefy webhook delivery.
///
/// Deliveries are signed `X-Forgefy-Signature: sha256=<hex>` — an HMAC-SHA256
/// of the raw request body using the `webhookSecret` returned when the job was
/// created. Always verify against the RAW body bytes, before any JSON
/// parsing/re-serialization (which can reorder keys and break the signature).
///
/// [payload] may be the raw body as a `String` or as bytes (`List<int>`).
/// Comparison is constant-time. Returns `false` on any malformed input rather
/// than throwing, so a bad delivery is simply rejected.
bool verifySignature(Object payload, String? signatureHeader, String secret) {
  if (signatureHeader == null || signatureHeader.isEmpty || secret.isEmpty) {
    return false;
  }

  final given = signatureHeader.startsWith('sha256=')
      ? signatureHeader.substring(7)
      : signatureHeader;
  if (!RegExp(r'^[0-9a-fA-F]{64}$').hasMatch(given)) return false;

  final List<int> bytes;
  if (payload is String) {
    bytes = utf8.encode(payload);
  } else if (payload is List<int>) {
    bytes = payload;
  } else {
    throw ArgumentError('payload must be a String or List<int>');
  }

  final expected = Hmac(sha256, utf8.encode(secret)).convert(bytes).bytes;
  final provided = _hexDecode(given);
  return _constantTimeEquals(expected, provided);
}

List<int> _hexDecode(String hex) {
  final out = List<int>.filled(hex.length ~/ 2, 0);
  for (var i = 0; i < out.length; i++) {
    out[i] = int.parse(hex.substring(i * 2, i * 2 + 2), radix: 16);
  }
  return out;
}

bool _constantTimeEquals(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  var diff = 0;
  for (var i = 0; i < a.length; i++) {
    diff |= a[i] ^ b[i];
  }
  return diff == 0;
}
