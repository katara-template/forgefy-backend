# Forgefy — Functionality Assessment

**Date:** 2026-07-30
**Overall Score: ~86% launch-ready** (was ~58% on 2026-05-27)

Scores below reflect what is **verified in code and by running the test suites**.
Anything depending on a live third-party call is marked ⚠️ **unverified** — those
need real credentials and a manual end-to-end run, not a code read.

---

## What changed since the 2026-05-27 assessment

All five roadmap priorities from the previous revision are implemented:

| Roadmap item | Where |
|---|---|
| Build error surfacing | `app/core/build_errors.py` + `build_error` on project docs |
| Build log streaming | `app/api/ws/build_logs.py` |
| Recall.ai bot (Meet/Zoom/Teams) | `app/connectors/recall.py`, routed via `factory.py` |
| Blueprint inline editing | `PATCH /api/v1/voxa/blueprint/{id}` (`app/api/v1/blueprints.py`) |
| Deepgram reconnection | backoff in `app/transcription/deepgram_client.py`, `deepgram_live.py` |

Fixed in this revision:

- **CORS** — `create_app()` hardcoded `allow_origins=["*"]` with
  `allow_credentials=True`, ignoring the documented `CORS_ORIGINS` setting.
  Because Starlette reflects the caller's `Origin` when wildcard and credentials
  are combined, any site could issue credentialed cross-origin requests. Now
  driven by `CORS_ORIGINS`; a literal `*` forces credentials off. Pinned by
  `tests/test_cors.py`.
- **Admin dashboard Settings page** — was entirely local React state, silently
  discarding every change. Build model and API keys are now wired to the real
  endpoints. See "Known gaps" for the two sections that were removed.
- **CI was red on both repos** — 5 ruff errors in the backend; 2 stale frontend
  tests in `forgefy-meeting-to-app` (they predated `ensureFreshToken()`, whose
  extra `fetch` shifted their mock chains). Both green now.
- **`forgefy-dashboard` had no CI and no typecheck script** — both added.

---

## Feature Breakdown

| Feature | % | Notes |
|---|---|---|
| **Auth (register / login / Google)** | 90% | JWT + Firebase. Missing: email verify, password reset |
| **Dashboard UI** | 85% | WebSocket updates and session/project cards work |
| **Session creation (physical upload)** | 80% | Upload → transcription → blueprint complete. ⚠️ Deepgram needs a live run |
| **Session creation (Meet / Zoom / Teams)** | 65% | Recall.ai wired for all three. ⚠️ Bot actually joining is unverified |
| **Transcription (Deepgram)** | 75% | Reconnection with backoff now implemented. ⚠️ Live stream unverified |
| **Feature extraction pipeline** | 70% | LangGraph chain + short-fragment guard. Silent fallback synthesis still passes weak blueprints through |
| **Blueprint generation + approval** | 90% | Aggregation solid; inline editing now available |
| **Build agent (file writing)** | 70% | Claude tool loop works; 60-iteration cap can still truncate very large apps |
| **GitHub push** | 85% | Standard REST API, token handling correct |
| **Cloudflare Pages deploy (Next.js)** | 60% | ⚠️ Depends on valid credentials + `next.config.js` patch holding |
| **Appetize preview (Flutter)** | 55% | ⚠️ APK build in Docker is the risky step |
| **Expo Snack preview (React Native)** | 45% | ⚠️ Expo Snack API is flaky; file size limits bite on large projects |
| **Project editor (prompt → update)** | 80% | Update agent + GitHub push; failures now surface via `build_error` |
| **GitHub OAuth linking** | 70% | ⚠️ Needs real `GITHUB_CLIENT_ID/SECRET` to verify |
| **Developer API (keys / extract / usage)** | 85% | Key hashing, quotas, async jobs, webhooks with HMAC signing |
| **Admin dashboard** | 80% | All pages wired to the real API |

---

## Verification status

| Check | Result |
|---|---|
| `pytest tests/` | **419 passed, 0 failed** (~17 min) |
| `ruff check app/ tests/` | clean |
| `forgefy-meeting-to-app` — typecheck / test / build | clean · **13 passed** · clean |
| `forgefy-dashboard` — lint / typecheck / build | clean (6 pre-existing shadcn warnings) · clean · clean |

---

## Known gaps

### 1. Live third-party paths are unproven
Recall.ai joining a real meeting, Deepgram streaming, and the three preview/deploy
targets (Cloudflare, Appetize, Expo Snack) have **never been verified end-to-end**.
This is the single largest risk to the score above — every one of them is code-complete
but credential-dependent. Physical upload remains the safest demo path.

### 2. Outbound webhook subscriptions have no backend
The admin Settings page previously showed a webhook CRUD UI backed by nothing.
`app/api/v1/webhooks.py` handles **inbound** webhooks only (Recall, NotchPay);
per-job delivery webhooks exist for the async extract API. There is no
platform-wide webhook subscription model. The fake UI was removed rather than
left to silently discard input — building the real feature is a scoped piece of
work, not a launch fix.

### 3. Notification preferences have no backend
Same story — the toggles persisted nothing and were removed.

### 4. Silent fallback synthesis in extraction
If the LangGraph chain degrades, a weak blueprint can still pass through without
signalling low confidence. Worth a confidence score on the blueprint.

### 5. Test suite runtime
419 tests in ~17 minutes (~2.5s each) because several exercise real retry/backoff
timing. Injecting a clock into the retry helpers would cut this substantially.

### 6. `src/lib/mock-data.ts` (dashboard) is vestigial
Only its `Status` type is still imported (`status-badge.tsx`, `_auth/index.tsx`).
The exported fixture data is dead and should be deleted once the type moves.

---

## Roadmap to ~95%

| Action | Who | Effort | % Gain |
|---|---|---|---|
| Verify Recall.ai bot end-to-end | Manual + `RECALL_API_KEY`, `PUBLIC_API_BASE_URL` | Low | +4% |
| Verify Deepgram live streaming | Manual + credentials | Low | +2% |
| Verify the 3 deploy/preview targets | Manual + credentials | Medium | +3% |
| Blueprint confidence score (kill silent fallback) | Dev | Medium | +2% |
| Email verification + password reset | Dev | Medium | +2% |
| Outbound webhook subscriptions (backend + UI) | Dev | Medium | — feature |
