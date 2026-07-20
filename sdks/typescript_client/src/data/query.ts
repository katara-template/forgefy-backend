/**
 * A PostgREST query builder — the data layer for Supabase and Neon.
 *
 * Chain filters and modifiers, then `await` the builder (it's a thenable) or
 * call {@link execute}:
 *
 * ```ts
 * const rows = await client
 *   .from<Todo>("todos")
 *   .select("id, title, done")
 *   .eq("user_id", userId)
 *   .order("created_at", { ascending: false })
 *   .limit(20);
 *
 * await client.from("todos").insert({ title: "Ship SDK" });
 * const row = await client.from<Todo>("todos").insert(t).single();
 * ```
 */

import type { ForgefyHttp } from "../http.js";

type Row = Record<string, unknown>;

/**
 * `Result` defaults to `T[]`; {@link single} re-types it to `T | null`. The
 * request fires when the builder is first awaited/`.then`-ed, exactly once.
 */
export class ForgefyQuery<T extends Row = Row, Result = T[]> implements PromiseLike<Result> {
  private method = "GET";
  private body: unknown;
  private readonly params: string[] = [];
  private readonly headers: Record<string, string> = {};
  private singleRow = false;
  private pending: Promise<Result> | undefined;

  constructor(
    private readonly http: ForgefyHttp,
    private readonly restUrl: string,
    private readonly table: string,
  ) {}

  // ── Terminal verbs ─────────────────────────────────────────────────────────

  /** Read rows. `columns` is a PostgREST select list, e.g. "id, title" or "*, author(name)". */
  select(columns = "*"): this {
    this.method = "GET";
    this.params.push(`select=${encodeURIComponent(columns)}`);
    return this;
  }

  /** Insert one row or an array of rows. Returns the inserted rows. */
  insert(values: Row | Row[]): this {
    this.method = "POST";
    this.body = values;
    this.headers.Prefer = "return=representation";
    return this;
  }

  /** Insert-or-update on conflict. `onConflict` names the unique column(s). */
  upsert(values: Row | Row[], opts: { onConflict?: string } = {}): this {
    this.method = "POST";
    this.body = values;
    this.headers.Prefer = "return=representation,resolution=merge-duplicates";
    if (opts.onConflict) this.params.push(`on_conflict=${encodeURIComponent(opts.onConflict)}`);
    return this;
  }

  /** Patch rows matching the filters (PostgREST refuses an unfiltered update). */
  update(values: Row): this {
    this.method = "PATCH";
    this.body = values;
    this.headers.Prefer = "return=representation";
    return this;
  }

  /** Delete rows matching the filters (PostgREST refuses an unfiltered delete). */
  delete(): this {
    this.method = "DELETE";
    this.headers.Prefer = "return=representation";
    return this;
  }

  // ── Filters ────────────────────────────────────────────────────────────────

  eq(column: string, value: unknown): this {
    return this.filter(column, "eq", value);
  }
  neq(column: string, value: unknown): this {
    return this.filter(column, "neq", value);
  }
  gt(column: string, value: unknown): this {
    return this.filter(column, "gt", value);
  }
  gte(column: string, value: unknown): this {
    return this.filter(column, "gte", value);
  }
  lt(column: string, value: unknown): this {
    return this.filter(column, "lt", value);
  }
  lte(column: string, value: unknown): this {
    return this.filter(column, "lte", value);
  }
  /** SQL LIKE (case-sensitive); use `%` as the wildcard. */
  like(column: string, pattern: string): this {
    return this.filter(column, "like", pattern);
  }
  /** SQL ILIKE (case-insensitive). */
  ilike(column: string, pattern: string): this {
    return this.filter(column, "ilike", pattern);
  }
  /** `column IS value`, for null / true / false. */
  is(column: string, value: null | boolean): this {
    return this.filter(column, "is", value === null ? "null" : value);
  }
  /** `column IN (values)`. */
  in(column: string, values: unknown[]): this {
    return this.filter(column, "in", `(${values.join(",")})`);
  }

  // ── Modifiers ──────────────────────────────────────────────────────────────

  order(column: string, opts: { ascending?: boolean } = {}): this {
    const dir = opts.ascending === false ? "desc" : "asc";
    this.params.push(`order=${encodeURIComponent(`${column}.${dir}`)}`);
    return this;
  }

  limit(count: number): this {
    this.params.push(`limit=${count}`);
    return this;
  }

  /** Zero-based inclusive row window (like SQL OFFSET/LIMIT). */
  range(from: number, to: number): this {
    this.params.push(`offset=${from}`, `limit=${to - from + 1}`);
    return this;
  }

  /** Return a single object instead of a list, and error if not exactly one row. */
  single(): ForgefyQuery<T, T | null> {
    this.singleRow = true;
    this.headers.Accept = "application/vnd.pgrst.object+json";
    return this as unknown as ForgefyQuery<T, T | null>;
  }

  // ── Execution ──────────────────────────────────────────────────────────────

  /** Fire the request (idempotent — repeated awaits reuse the same promise). */
  execute(): Promise<Result> {
    return (this.pending ??= this.run());
  }

  then<TResult1 = Result, TResult2 = never>(
    onfulfilled?: ((value: Result) => TResult1 | PromiseLike<TResult1>) | null,
    onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
  ): Promise<TResult1 | TResult2> {
    return this.execute().then(onfulfilled, onrejected);
  }

  catch<TResult = never>(
    onrejected?: ((reason: unknown) => TResult | PromiseLike<TResult>) | null,
  ): Promise<Result | TResult> {
    return this.execute().catch(onrejected);
  }

  private async run(): Promise<Result> {
    const query = this.params.length === 0 ? "" : `?${this.params.join("&")}`;
    const res = await this.http.send(this.method, `${this.restUrl}/${this.table}${query}`, {
      body: this.body,
      headers: this.headers,
      // GET is safe to retry; writes carry no idempotency key, so are not.
      retryOn5xx: this.method === "GET",
    });
    // With `Prefer: return=representation` PostgREST replies with a list even
    // for single-row writes; `.single()` unwraps it.
    if (this.singleRow && Array.isArray(res.data)) {
      return (res.data.length === 0 ? null : res.data[0]) as Result;
    }
    return res.data as Result;
  }

  private filter(column: string, op: string, value: unknown): this {
    this.params.push(`${encodeURIComponent(column)}=${encodeURIComponent(`${op}.${String(value)}`)}`);
    return this;
  }
}
