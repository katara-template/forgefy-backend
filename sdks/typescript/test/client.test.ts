import { createHmac } from "node:crypto";
import { describe, expect, it, vi } from "vitest";

import {
  AuthenticationError,
  Forgefy,
  JobFailedError,
  JobTimeoutError,
  QuotaExceededError,
  RateLimitError,
  ValidationError,
  verifySignature,
} from "../src/index.js";

const BASE = "https://api.example.com";

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

function problem(status: number, detail: string): Response {
  return jsonResponse(status, { type: "about:blank", title: "err", status, detail });
}

const EXTRACT_OK = {
  id: "e1",
  model_tier: "standard",
  features: [{ title: "OAuth login", description: "SSO", priority: "high" }],
  questions: [],
  conflicts: [],
  action_items: [],
  usage: { input_tokens: 100, output_tokens: 40 },
};

function makeClient(fetchMock: typeof fetch, opts: Partial<ConstructorParameters<typeof Forgefy>[0]> = {}) {
  return new Forgefy({
    apiKey: "fgy_live_test",
    baseUrl: BASE,
    fetch: fetchMock,
    retryDelayMs: 1, // keep retry tests fast
    ...opts,
  });
}

describe("extract", () => {
  it("sends the key, body, and parses the result", async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, EXTRACT_OK));
    const client = makeClient(fetchMock as unknown as typeof fetch);

    const result = await client.extract({
      transcript: "we need oauth",
      extractors: ["features"],
      modelTier: "standard",
    });

    expect(result.features[0].title).toBe("OAuth login");
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${BASE}/api/v1/extract`);
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer fgy_live_test");
    expect(JSON.parse(init.body as string)).toEqual({
      transcript: "we need oauth",
      extractors: ["features"],
      model_tier: "standard",
    });
  });

  it("omits optional fields it was not given", async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, EXTRACT_OK));
    const client = makeClient(fetchMock as unknown as typeof fetch);

    await client.extract({ transcript: "hello world" });

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toEqual({ transcript: "hello world" });
  });

  it("maps problem+json to typed errors", async () => {
    const cases: Array<[Response, unknown]> = [
      [problem(401, "bad key"), AuthenticationError],
      [problem(402, "quota exhausted, resets on August 1"), QuotaExceededError],
      [problem(422, "transcript too long"), ValidationError],
    ];
    for (const [response, errorClass] of cases) {
      const client = makeClient(vi.fn(async () => response) as unknown as typeof fetch);
      await expect(client.extract({ transcript: "t" })).rejects.toBeInstanceOf(errorClass as never);
    }
  });

  it("retries 429 then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(problem(429, "slow down"))
      .mockResolvedValueOnce(jsonResponse(200, EXTRACT_OK));
    const client = makeClient(fetchMock as unknown as typeof fetch);

    const result = await client.extract({ transcript: "t" });
    expect(result.id).toBe("e1");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("gives up with RateLimitError after maxRetries", async () => {
    const fetchMock = vi.fn(async () => problem(429, "slow down"));
    const client = makeClient(fetchMock as unknown as typeof fetch, { maxRetries: 1 });

    await expect(client.extract({ transcript: "t" })).rejects.toBeInstanceOf(RateLimitError);
    expect(fetchMock).toHaveBeenCalledTimes(2); // initial + 1 retry
  });

  it("does NOT retry the sync extract on 5xx (tokens would double-bill)", async () => {
    const fetchMock = vi.fn(async () => problem(502, "all extractors errored"));
    const client = makeClient(fetchMock as unknown as typeof fetch);

    await expect(client.extract({ transcript: "t" })).rejects.toMatchObject({ status: 502 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("jobs", () => {
  it("create auto-generates an Idempotency-Key and retries 5xx safely", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(problem(502, "queue hiccup"))
      .mockResolvedValueOnce(jsonResponse(202, { job_id: "j1", status: "queued", webhook_secret: null }));
    const client = makeClient(fetchMock as unknown as typeof fetch);

    const job = await client.jobs.create({ transcript: "long one" });

    expect(job.job_id).toBe("j1");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const keys = fetchMock.mock.calls.map(
      (c) => ((c as unknown as [string, RequestInit])[1].headers as Record<string, string>)["Idempotency-Key"],
    );
    expect(keys[0]).toBeTruthy();
    expect(keys[0]).toBe(keys[1]); // the retry reuses the same key — no double job
  });

  it("create passes an explicit idempotency key and webhook url through", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(202, { job_id: "j1", status: "queued", webhook_secret: "s3cret" }),
    );
    const client = makeClient(fetchMock as unknown as typeof fetch);

    const job = await client.jobs.create({
      transcript: "t",
      webhookUrl: "https://example.com/hook",
      idempotencyKey: "import-42",
    });

    expect(job.webhook_secret).toBe("s3cret");
    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect((init.headers as Record<string, string>)["Idempotency-Key"]).toBe("import-42");
    expect(JSON.parse(init.body as string).webhook_url).toBe("https://example.com/hook");
  });

  it("waitFor polls until done", async () => {
    const statuses = ["queued", "processing", "done"];
    let call = 0;
    const fetchMock = vi.fn(async () =>
      jsonResponse(200, {
        job_id: "j1",
        status: statuses[call++],
        model_tier: "standard",
        created_at: "2026-07-16T00:00:00Z",
        result: call === 3 ? { features: [], usage: { input_tokens: 1, output_tokens: 1 } } : null,
        error: null,
      }),
    );
    const client = makeClient(fetchMock as unknown as typeof fetch);

    const job = await client.jobs.waitFor("j1", { pollIntervalMs: 1 });
    expect(job.status).toBe("done");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("waitFor throws JobFailedError on failed", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(200, {
        job_id: "j1",
        status: "failed",
        model_tier: "standard",
        created_at: "2026-07-16T00:00:00Z",
        result: null,
        error: "provider down",
      }),
    );
    const client = makeClient(fetchMock as unknown as typeof fetch);

    await expect(client.jobs.waitFor("j1", { pollIntervalMs: 1 })).rejects.toBeInstanceOf(JobFailedError);
  });

  it("waitFor times out", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(200, {
        job_id: "j1",
        status: "processing",
        model_tier: "standard",
        created_at: "2026-07-16T00:00:00Z",
        result: null,
        error: null,
      }),
    );
    const client = makeClient(fetchMock as unknown as typeof fetch);

    await expect(client.jobs.waitFor("j1", { pollIntervalMs: 1, timeoutMs: 5 })).rejects.toBeInstanceOf(
      JobTimeoutError,
    );
  });
});

describe("usage", () => {
  it("fetches the quota snapshot", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(200, {
        tier: "starter",
        tier_name: "Starter",
        monthly_tokens: 5_000_000,
        tokens_used: 1_200_000,
        tokens_remaining: 3_800_000,
        resets_at: "2026-08-01T00:00:00+00:00",
      }),
    );
    const client = makeClient(fetchMock as unknown as typeof fetch);

    const usage = await client.usage();
    expect(usage.tokens_remaining).toBe(3_800_000);
    const [url] = fetchMock.mock.calls[0] as unknown as [string];
    expect(url).toBe(`${BASE}/api/v1/usage`);
  });
});

describe("verifySignature", () => {
  const secret = "s3cret";
  const body = JSON.stringify({ type: "extract.job.completed", job_id: "j1" });
  const goodSig = `sha256=${createHmac("sha256", secret).update(body).digest("hex")}`;

  it("accepts a valid signature", () => {
    expect(verifySignature(body, goodSig, secret)).toBe(true);
  });

  it("accepts the bare hex form too", () => {
    expect(verifySignature(body, goodSig.slice(7), secret)).toBe(true);
  });

  it("rejects a tampered body", () => {
    expect(verifySignature(body + "x", goodSig, secret)).toBe(false);
  });

  it("rejects the wrong secret", () => {
    expect(verifySignature(body, goodSig, "other")).toBe(false);
  });

  it("rejects garbage headers without throwing", () => {
    expect(verifySignature(body, undefined, secret)).toBe(false);
    expect(verifySignature(body, "", secret)).toBe(false);
    expect(verifySignature(body, "sha256=nothex", secret)).toBe(false);
  });
});
