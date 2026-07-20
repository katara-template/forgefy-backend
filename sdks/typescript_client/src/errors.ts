/**
 * Typed errors surfaced by the Forgefy app-runtime SDK.
 *
 * Backends disagree on their error envelopes — GoTrue uses
 * `{error, error_description}` (newer: `{code, msg}`), PostgREST uses
 * `{message, details, hint, code}`, and some gateways return RFC 7807
 * `{type, title, detail}`. {@link errorFromResponse} normalises all three so
 * app code can `catch` by meaning, not by provider.
 */

export interface ForgefyErrorOptions {
  status?: number;
  code?: string;
  details?: string;
}

/** Base class for every failure surfaced by the SDK. */
export class ForgefyError extends Error {
  readonly status: number;
  readonly code: string | undefined;
  readonly details: string | undefined;

  constructor(message: string, opts: ForgefyErrorOptions = {}) {
    super(message);
    this.name = new.target.name;
    this.status = opts.status ?? 0;
    this.code = opts.code;
    this.details = opts.details;
  }
}

/** 400 / 422 — the request was malformed or failed validation. */
export class ValidationError extends ForgefyError {}

/** 401 / 403 — not signed in, token expired, or blocked by row-level security. */
export class AuthError extends ForgefyError {}

/** 404 — the resource doesn't exist or is hidden by access rules. */
export class NotFoundError extends ForgefyError {}

/** 409 — a conflict, most often a unique-constraint violation on insert/upsert. */
export class ConflictError extends ForgefyError {}

/** 429 — rate limited. `Retry-After` is honoured (the SDK retries it once). */
export class RateLimitError extends ForgefyError {}

/** 5xx — the backend faulted. */
export class ServerError extends ForgefyError {}

/** The request never got an HTTP response (DNS, connection refused, timeout). */
export class ConnectionError extends ForgefyError {}

type ErrorBody = Record<string, unknown>;

/** Map an HTTP status + decoded body to the matching error subclass. */
export function errorFromResponse(status: number, body: unknown): ForgefyError {
  const map: ErrorBody = body !== null && typeof body === "object" ? (body as ErrorBody) : {};
  const message = messageFrom(map, status);
  const code = stringOf(map.code) ?? stringOf(map.error);
  const details = stringOf(map.details) ?? stringOf(map.hint);
  const opts: ForgefyErrorOptions = { status, code, details };

  switch (status) {
    case 400:
    case 422:
      return new ValidationError(message, opts);
    case 401:
    case 403:
      return new AuthError(message, opts);
    case 404:
      return new NotFoundError(message, opts);
    case 409:
      return new ConflictError(message, opts);
    case 429:
      return new RateLimitError(message, opts);
    default:
      return status >= 500 ? new ServerError(message, opts) : new ForgefyError(message, opts);
  }
}

function messageFrom(body: ErrorBody, status: number): string {
  const candidate =
    stringOf(body.error_description) ?? // GoTrue (older)
    stringOf(body.msg) ?? // GoTrue (newer)
    stringOf(body.message) ?? // PostgREST
    stringOf(body.detail) ?? // RFC 7807
    stringOf(body.title) ?? // RFC 7807
    stringOf(body.error); // last resort (may be a code)
  return candidate && candidate.length > 0 ? candidate : `Request failed with status ${status}`;
}

function stringOf(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}
