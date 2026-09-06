import { randomUUID } from "node:crypto";

import {
  APIConnectionError,
  errorFromProblem,
  JobFailedError,
  JobTimeoutError,
  type ProblemDetails,
} from "./errors.js";
import type {
  CreateJobParams,
  ExtractParams,
  ExtractResult,
  JobCreated,
  JobStatusResponse,
  UsageResponse,
} from "./types.js";

export interface ForgefyOptions {
  /** API key from the Forgefy dashboard's Developers page (fgy_live_…). */
  apiKey: string;
  /** API origin, e.g. "https://api.your-forgefy.com". Defaults to $FORGEFY_API_URL, then localhost. */
  baseUrl?: string;
  /** Retries for rate-limited/transient failures. Default 2. */
  maxRetries?: number;
  /** Per-request timeout. Default 120s (sync extraction runs 4 model calls). */
  timeoutMs?: number;
  /** Injectable fetch, mainly for tests. Defaults to globalThis.fetch. */
  fetch?: typeof fetch;
  /** Base backoff between retries. Exposed for tests; leave default in production. */
  retryDelayMs?: number;
}

interface RequestOptions {
  body?: unknown;
  headers?: Record<string, string>;
  /**
   * Whether 5xx responses may be retried. GETs and idempotent job creation
   * are; the sync /extract POST is not (a retry would bill tokens twice for
   * work that may have partially run).
   */
  retryOn5xx?: boolean;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export class Forgefy {
  readonly jobs: Jobs;

  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly maxRetries: number;
  private readonly timeoutMs: number;
  private readonly retryDelayMs: number;
  private readonly fetchFn: typeof fetch;

  constructor(options: ForgefyOptions) {
    if (!options.apiKey) throw new Error("Forgefy: apiKey is required");
    this.apiKey = options.apiKey;
    this.baseUrl = (
      options.baseUrl ??
      globalThis.process?.env?.FORGEFY_API_URL ??
      "https://server.forgefy.dev"
    ).replace(/\/$/, "");
    this.maxRetries = options.maxRetries ?? 2;
    this.timeoutMs = options.timeoutMs ?? 120_000;
    this.retryDelayMs = options.retryDelayMs ?? 500;
    // Bind to the global object: browsers throw "Illegal invocation" if
    // window.fetch is called with any other `this` (as it would be when
    // stored on and invoked from this instance). A caller-supplied fetch is
    // used as-is — they're responsible for its binding.
    const resolvedFetch = options.fetch ?? globalThis.fetch;
    if (!resolvedFetch) {
      throw new Error(
        "Forgefy: no fetch available — pass options.fetch (Node <18 or a non-fetch environment)",
      );
    }
    this.fetchFn = options.fetch ? resolvedFetch : resolvedFetch.bind(globalThis);
    this.jobs = new Jobs(this);
  }

  /**
   * Synchronously extract features / questions / conflicts / action items
   * from a transcript (≤50k chars). For longer transcripts use jobs.create().
   */
  async extract(params: ExtractParams): Promise<ExtractResult> {
    return this.request<ExtractResult>("POST", "/api/v1/extract", {
      body: buildExtractBody(params),
    });
  }

  /** The key owner's tier, monthly token budget, consumption, and reset date. */
  async usage(): Promise<UsageResponse> {
    return this.request<UsageResponse>("GET", "/api/v1/usage", { retryOn5xx: true });
  }

  /** @internal */
  async request<T>(method: string, path: string, opts: RequestOptions = {}): Promise<T> {
    for (let attempt = 0; ; attempt++) {
      let response: Response;
      try {
        response = await this.send(method, path, opts);
      } catch (err) {
        if (attempt < this.maxRetries) {
          await sleep(this.backoff(attempt));
          continue;
        }
        throw new APIConnectionError(
          `Could not reach the Forgefy API at ${this.baseUrl}: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      if (response.ok) {
        return (response.status === 204 ? undefined : await response.json()) as T;
      }

      const retryable = response.status === 429 || (opts.retryOn5xx === true && response.status >= 500);
      if (retryable && attempt < this.maxRetries) {
        const retryAfter = Number(response.headers.get("retry-after"));
        await sleep(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : this.backoff(attempt));
        continue;
      }

      const problem = (await response.json().catch(() => ({}))) as ProblemDetails;
      throw errorFromProblem(response.status, problem);
    }
  }

  private async send(method: string, path: string, opts: RequestOptions): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      return await this.fetchFn(`${this.baseUrl}${path}`, {
        method,
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
          "User-Agent": "forgefy-sdk-ts/0.1.1",
          ...opts.headers,
        },
        body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  private backoff(attempt: number): number {
    // 1x, 2x, 4x… the base delay, with a little jitter.
    return this.retryDelayMs * 2 ** attempt * (0.8 + Math.random() * 0.4);
  }
}

export interface WaitForOptions {
  /** How often to poll. Default 3s. */
  pollIntervalMs?: number;
  /** Give up (JobTimeoutError) after this long. Default 10 min. */
  timeoutMs?: number;
}

class Jobs {
  constructor(private readonly client: Forgefy) {}

  /**
   * Queue an async extraction job (transcripts up to 200k chars).
   *
   * An Idempotency-Key is generated automatically so network-level retries
   * can never run the same job twice; pass your own to dedupe across
   * processes (e.g. a stable import-batch id).
   */
  async create(params: CreateJobParams): Promise<JobCreated> {
    return this.client.request<JobCreated>("POST", "/api/v1/extract/jobs", {
      body: {
        ...buildExtractBody(params),
        ...(params.webhookUrl !== undefined ? { webhook_url: params.webhookUrl } : {}),
      },
      headers: { "Idempotency-Key": params.idempotencyKey ?? randomUUID() },
      retryOn5xx: true, // safe: the idempotency key dedupes on the server
    });
  }

  /** Current status of a job; `result` is set once status is "done". */
  async get(jobId: string): Promise<JobStatusResponse> {
    return this.client.request<JobStatusResponse>(
      "GET",
      `/api/v1/extract/jobs/${encodeURIComponent(jobId)}`,
      { retryOn5xx: true },
    );
  }

  /**
   * Poll a job until it finishes. Resolves with the "done" status response,
   * throws JobFailedError on "failed" and JobTimeoutError on deadline.
   */
  async waitFor(jobId: string, options: WaitForOptions = {}): Promise<JobStatusResponse> {
    const pollIntervalMs = options.pollIntervalMs ?? 3_000;
    const timeoutMs = options.timeoutMs ?? 600_000;
    const deadline = Date.now() + timeoutMs;

    for (;;) {
      const job = await this.get(jobId);
      if (job.status === "done") return job;
      if (job.status === "failed") throw new JobFailedError(jobId, job.error ?? undefined);
      if (Date.now() + pollIntervalMs > deadline) throw new JobTimeoutError(jobId, timeoutMs);
      await sleep(pollIntervalMs);
    }
  }
}

function buildExtractBody(params: ExtractParams): Record<string, unknown> {
  return {
    transcript: params.transcript,
    ...(params.extractors !== undefined ? { extractors: params.extractors } : {}),
    ...(params.modelTier !== undefined ? { model_tier: params.modelTier } : {}),
  };
}
