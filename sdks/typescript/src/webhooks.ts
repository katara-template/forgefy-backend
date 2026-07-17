import { createHmac, timingSafeEqual } from "node:crypto";

/**
 * Verify a Forgefy webhook delivery.
 *
 * Deliveries are signed `X-Forgefy-Signature: sha256=<hex>` — an HMAC-SHA256
 * of the raw request body using the `webhook_secret` returned when the job
 * was created. Always verify against the RAW body bytes, before any JSON
 * parsing/re-serialization (which can reorder keys and break the signature).
 *
 * Comparison is constant-time.
 */
export function verifySignature(
  payload: string | Uint8Array,
  signatureHeader: string | null | undefined,
  secret: string,
): boolean {
  if (!signatureHeader || !secret) return false;

  const given = signatureHeader.startsWith("sha256=") ? signatureHeader.slice(7) : signatureHeader;
  if (!/^[0-9a-f]{64}$/i.test(given)) return false;

  const expected = createHmac("sha256", secret).update(payload).digest();
  const provided = Buffer.from(given, "hex");
  return provided.length === expected.length && timingSafeEqual(expected, provided);
}
