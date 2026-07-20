import { describe, expect, it, vi } from "vitest";

import {
  AuthError,
  ConflictError,
  ForgefyClient,
  InMemorySessionStore,
  persistentSessionStore,
  type SessionStore,
} from "../src/index.js";

const URL = "https://proj.supabase.co";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** A fetch mock that records every call and replies via `handler`. */
function harness(
  handler: (url: string, init: RequestInit) => Response,
  opts: { sessionStore?: SessionStore } = {},
) {
  const calls: { url: string; init: RequestInit }[] = [];
  const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const resolved = init ?? {};
    calls.push({ url, init: resolved });
    return handler(url, resolved);
  }) as unknown as typeof fetch;

  const client = new ForgefyClient({
    url: URL,
    anonKey: "anon-key",
    fetch: fetchMock,
    ...opts,
  });
  return { client, calls };
}

const headerOf = (init: RequestInit, name: string) =>
  (init.headers as Record<string, string>)[name];

describe("data query building", () => {
  it("select with filters, order and limit → correct URL and verb", async () => {
    const h = harness(() => jsonResponse([{ id: 1, title: "a" }]));

    const rows = await h.client
      .from("todos")
      .select("id, title")
      .eq("done", false)
      .order("created_at", { ascending: false })
      .limit(20);

    expect(Array.isArray(rows)).toBe(true);
    const { url, init } = h.calls[0]!;
    expect(init.method).toBe("GET");
    expect(url).toContain("/rest/v1/todos?");
    expect(url).toContain("select=id%2C%20title");
    expect(url).toContain("done=eq.false");
    expect(url).toContain("order=created_at.desc");
    expect(url).toContain("limit=20");
  });

  it("insert → POST with return=representation and body", async () => {
    const h = harness(() => jsonResponse([{ id: 7, title: "Ship SDK" }]));
    await h.client.from("todos").insert({ title: "Ship SDK" });

    const { init } = h.calls[0]!;
    expect(init.method).toBe("POST");
    expect(headerOf(init, "Prefer")).toContain("return=representation");
    expect(JSON.parse(init.body as string)).toEqual({ title: "Ship SDK" });
  });

  it("update with filter → PATCH", async () => {
    const h = harness(() => jsonResponse([]));
    await h.client.from("todos").update({ done: true }).eq("id", 7);

    const { url, init } = h.calls[0]!;
    expect(init.method).toBe("PATCH");
    expect(url).toContain("id=eq.7");
  });

  it("single() unwraps the representation list to one object", async () => {
    const h = harness(() => jsonResponse([{ id: 7, title: "a" }]));
    const row = await h.client.from("todos").insert({ title: "a" }).single();
    expect(row).toEqual({ id: 7, title: "a" });
  });

  it("anon key is sent before sign-in", async () => {
    const h = harness(() => jsonResponse([]));
    await h.client.from("todos").select();

    const { init } = h.calls[0]!;
    expect(headerOf(init, "apikey")).toBe("anon-key");
    expect(headerOf(init, "Authorization")).toBe("Bearer anon-key");
  });
});

describe("auth", () => {
  it("signIn parses the session and later queries carry the user token", async () => {
    const h = harness((url) => {
      if (url.includes("/token")) {
        return jsonResponse({
          access_token: "user-jwt",
          refresh_token: "refresh-1",
          expires_in: 3600,
          user: { id: "u1", email: "a@b.c" },
        });
      }
      return jsonResponse([]);
    });

    const session = await h.client.auth.signInWithPassword({ email: "a@b.c", password: "pw" });
    expect(session.accessToken).toBe("user-jwt");
    expect(h.client.auth.currentUser?.id).toBe("u1");

    await h.client.from("todos").select();
    const query = h.calls[h.calls.length - 1]!;
    expect(headerOf(query.init, "Authorization")).toBe("Bearer user-jwt");
    expect(headerOf(query.init, "apikey")).toBe("anon-key");
  });

  it("onAuthStateChange emits signedIn then signedOut", async () => {
    const h = harness((url) => {
      if (url.includes("/token")) {
        return jsonResponse({
          access_token: "user-jwt",
          refresh_token: "r",
          expires_in: 3600,
          user: { id: "u1" },
        });
      }
      return jsonResponse({});
    });

    const events: string[] = [];
    h.client.auth.onAuthStateChange((c) => events.push(c.event));

    await h.client.auth.signInWithPassword({ email: "a@b.c", password: "pw" });
    await h.client.auth.signOut();

    expect(events).toEqual(["signedIn", "signedOut"]);
    expect(h.client.auth.currentSession).toBeNull();
  });

  it("restoreSession refreshes an expired stored session", async () => {
    const store = new InMemorySessionStore();
    store.write(
      JSON.stringify({
        access_token: "old",
        refresh_token: "refresh-1",
        expires_at: 0, // long past
        user: { id: "u1" },
      }),
    );

    let refreshCalls = 0;
    const h = harness(
      (url) => {
        if (url.includes("grant_type=refresh_token")) {
          refreshCalls++;
          return jsonResponse({
            access_token: "fresh-jwt",
            refresh_token: "refresh-2",
            expires_in: 3600,
            user: { id: "u1" },
          });
        }
        return jsonResponse({});
      },
      { sessionStore: store },
    );

    const restored = await h.client.auth.restoreSession();
    expect(refreshCalls).toBe(1);
    expect(restored?.accessToken).toBe("fresh-jwt");
  });
});

describe("errors", () => {
  it("401 → AuthError", async () => {
    const h = harness(() => jsonResponse({ msg: "invalid token" }, 401));
    const err = await h.client.from("todos").select().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(AuthError);
    expect((err as AuthError).status).toBe(401);
    expect((err as AuthError).message).toBe("invalid token");
  });

  it("409 unique violation → ConflictError with PostgREST fields", async () => {
    const h = harness(() =>
      jsonResponse(
        {
          message: "duplicate key value violates unique constraint",
          code: "23505",
          details: "Key (email)=(a@b.c) already exists.",
        },
        409,
      ),
    );

    const err = await h.client.from("users").insert({ email: "a@b.c" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ConflictError);
    expect((err as ConflictError).code).toBe("23505");
    expect((err as ConflictError).details).toContain("already exists");
  });
});

describe("session store", () => {
  it("persistentSessionStore adapts a KeyValue storage", async () => {
    const backing = new Map<string, string>();
    const store = persistentSessionStore({
      getItem: (k) => backing.get(k) ?? null,
      setItem: (k, v) => void backing.set(k, v),
      removeItem: (k) => void backing.delete(k),
    });
    await store.write("hello");
    expect(await store.read()).toBe("hello");
    await store.delete();
    expect(await store.read()).toBeNull();
  });
});
