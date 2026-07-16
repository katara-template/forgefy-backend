/** Typed errors mapped from the API's RFC 7807 problem+json responses. */

export interface ProblemDetails {
  type?: string;
  title?: string;
  status?: number;
  detail?: string;
  instance?: string;
}

export class ForgefyError extends Error {
  readonly status: number;
  readonly detail: string | undefined;
  readonly problemType: string | undefined;

  constructor(message: string, opts: { status?: number; detail?: string; problemType?: string } = {}) {
    super(message);
    this.name = new.target.name;
    this.status = opts.status ?? 0;
    this.detail = opts.detail;
    this.problemType = opts.problemType;
  }
}
/** 401 — missing, malformed, or revoked API key. */
export class AuthenticationError extends ForgefyError {}

/** 402 — monthly token budget exhausted (free tier). `detail` includes the reset date. */
export class QuotaExceededError extends ForgefyError {}

/** 404 — the resource doesn't exist or belongs to another account. */
export class NotFoundError extends ForgefyError {}

/** 422 — invalid request (empty/oversized transcript, bad extractor name, …). */
export class ValidationError extends ForgefyError {}

/** 429 — over 60 req/min for this key, or the server is at sync capacity. */
export class RateLimitError extends ForgefyError {}

/** 5xx — extraction failed on every model, queue unreachable, or other server fault. */
export class ServerError extends ForgefyError {}

/** The request never got an HTTP response (DNS, refused, timeout). */
export class APIConnectionError extends ForgefyError {}

/** jobs.waitFor: the job finished with status "failed". */
export class JobFailedError extends ForgefyError {
  readonly jobId: string;

  constructor(jobId: string, detail: string | undefined) {
    super(`Extract job ${jobId} failed${detail ? `: ${detail}` : ""}`, { detail });
    this.jobId = jobId;
  }
}

/** jobs.waitFor: the deadline passed before the job finished. */
export class JobTimeoutError extends ForgefyError {
  readonly jobId: string;

  constructor(jobId: string, timeoutMs: number) {
    super(`Timed out after ${timeoutMs}ms waiting for extract job ${jobId}`);
    this.jobId = jobId;
  }
}

export function errorFromProblem(status: number, problem: ProblemDetails): ForgefyError {
  const message = problem.detail || problem.title || `Forgefy API error ${status}`;
  const opts = { status, detail: problem.detail, problemType: problem.type };
  switch (status) {
    case 401:
      return new AuthenticationError(message, opts);
    case 402:
      return new QuotaExceededError(message, opts);
    case 404:
      return new NotFoundError(message, opts);
    case 422:
      return new ValidationError(message, opts);
    case 429:
      return new RateLimitError(message, opts);
    default:
      return status >= 500 ? new ServerError(message, opts) : new ForgefyError(message, opts);
  }
}
