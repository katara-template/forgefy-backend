/**
 * Low-level HTTP transport: header injection, retries, and error mapping.
 *
 * The retry/backoff logic mirrors the Forgefy Developer SDK's proven client —
 * the one piece every generated app used to hand-roll and get subtly wrong.
 */

import type { ResolvedConfig } from "./config.js";
import { ConnectionError, errorFromResponse } from "./errors.js";

const USER_AGENT = "forgefy-client-ts/0.1.0";

/** A decoded HTTP response. `headers` is retained for PostgREST `Content-Range`. */
export interface ForgefyResponse {
  status: number;
  data: unknown;
  headers: Headers;
}

export interface SendOptions {
  body?: unknown;
  headers?: Record<string, string>;
  /**
   * Whether 5xx may be retried — safe for reads and idempotent writes, never
   * for a plain non-idempotent POST.
   */
  retryOn5xx?: boolean;
}

/** Returns the current user access token, or null before sign-in. */
export type TokenProvider = () => string | null;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Shared transport used by both the auth and data layers so a single signed-in
 * session's token flows to every request.
 */
export class ForgefyHttp {
  /** Set by ForgefyClient to point at the live session's access token. */
  tokenProvider: TokenProvider | undefined;

  constructor(private readonly cfg: ResolvedConfig) {}

  async send(method: string, url: string, opts: SendOptions = {}): Promise<ForgefyResponse> {
    const token = this.tokenProvider?.() ?? this.cfg.anonKey;
    const headers: Record<string, string> = {
      ...(this.cfg.sendsApiKeyHeader ? { apikey: this.cfg.anonKey } : {}),
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "User-Agent": USER_AGENT,
      ...opts.headers,
    };

    for (let attempt = 0; ; attempt++) {
      let response: Response;
      try {
        response = await this.dispatch(method, url, headers, opts.body);
      } catch (err) {
        if (attempt < this.cfg.maxRetries) {
          await sleep(this.backoff(attempt));
          continue;
        }
        throw new ConnectionError(
          `Could not reach ${url}: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      if (response.ok) {
        return { status: response.status, data: await decode(response), headers: response.headers };
      }

      const retryable = response.status === 429 || (opts.retryOn5xx === true && response.status >= 500);
      if (retryable && attempt < this.cfg.maxRetries) {
        const retryAfter = Number(response.headers.get("retry-after"));
        await sleep(Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : this.backoff(attempt));
        continue;
      }

      throw errorFromResponse(response.status, await decode(response));
    }
  }

  private async dispatch(
    method: string,
    url: string,
    headers: Record<string, string>,
    body: unknown,
  ): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.cfg.timeoutMs);
    try {
      return await this.cfg.fetch(url, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  private backoff(attempt: number): number {
    // 1x, 2x, 4x… the base delay, with a little jitter.
    return this.cfg.retryDelayMs * 2 ** attempt * (0.8 + Math.random() * 0.4);
  }
}

async function decode(response: Response): Promise<unknown> {
  const text = await response.text();
  if (text.length === 0) return null;
  try {
    return JSON.parse(text);
  } catch {
    // Non-JSON body (e.g. a proxy's HTML error page) — surface it as text.
    return { message: text };
  }
}
