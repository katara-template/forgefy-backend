"""Build agent — Claude with file-system tools that implements a blueprint."""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic

from app.build.agent_tools import (
    _FINDING_SEVERITIES,
    REPORT_FINDINGS_TOOL,
    TOOLS,
    missing_report,
    reset_report,
    take_report,
)
from app.build.project_memory import update_project_memory_async

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 80
_WARN_AT_ITERATION = 50
# Cap on "you haven't written anything yet" pushbacks. Without a cap these
# consume every iteration when a model won't emit tool calls at all.
_MAX_NUDGES = 2
# Tools that only observe the workspace. Repeating one with identical arguments
# cannot produce a new answer until something mutates the workspace, so repeats
# are served a short notice instead of being re-executed — otherwise the agent
# can spend its whole iteration budget re-reading the same files.
_READ_ONLY_TOOLS = frozenset({
    "read_file", "list_files", "analyze_code", "grep", "glob",
})
# Tools that count as the agent actually producing code. edit_file belongs here
# alongside write_file: an agent that implements a change entirely through edits
# has done the work, and treating it as "no files written" would nudge it and
# then trip the exploration breaker on an agent that is making real progress.
_WRITE_TOOLS = frozenset({"write_file", "edit_file"})
# Consecutive tool-only turns without a write before we push the agent to act.
_MAX_EXPLORE_STREAK = 12

# Tool results are echoed back into context every turn, so they are capped.
# 1500 chars is ~375 tokens — smaller than a typical component file, so
# read_file (which advertises "the full text content") returned a fragment and
# the agent re-read the same file over and over trying to see the rest.
# Large-context backends can afford far more.
_TOOL_RESULT_LIMIT_SMALL_CTX = 1500
_TOOL_RESULT_LIMIT_LARGE_CTX = 24000

# Per-response output cap for the Anthropic loop. The old value (8096) was both a
# typo for 8192 and far too small: a single large write_file payload plus thinking
# tokens overran it and the file arrived truncated mid-generation. claude-sonnet-5
# reports max_tokens=128000 via the Models API; 32000 leaves generous headroom for
# a big file without letting one runaway turn spend the whole budget.
_MAX_OUTPUT_TOKENS = 32000
# Chat-completions backends (OpenAI, OpenRouter) are separate providers with their
# own, much lower ceilings — 16384 is the documented cap for gpt-4o-mini and a safe
# value across the OpenRouter chain.
_MAX_OUTPUT_TOKENS_CHAT = 16384

# Anthropic sliding window, in assistant/tool-result pairs. Claude's context is
# large, so this is generous — it exists to stop an 80-iteration build from
# resending every turn forever, not to squeeze into a small window.
_ANTHROPIC_HISTORY_PAIRS = 20
# Assistant/tool-result pairs after the first user message that are never
# trimmed. Everything up to the end of this anchor is byte-stable for the whole
# phase, which is what lets a cache breakpoint sit there and actually be read;
# measurement showed that once the window starts sliding, no breakpoint inside it
# can ever match a stored prefix again. Also keeps the opening of the task in
# view, which is where the requirements are.
_ANCHOR_TURNS = 3
_ANCHOR_END = 1 + _ANCHOR_TURNS * 2

_CACHE_CONTROL: dict[str, Any] = {"type": "ephemeral"}

# Adaptive thinking with a summarised display, so the build feed can narrate the
# agent's reasoning the way the Ollama path does. On claude-sonnet-5 thinking is
# already on by default and `display` only controls whether the summary text is
# returned, so this surfaces reasoning without buying extra thinking.
_THINKING_CONFIG: dict[str, Any] = {"type": "adaptive", "display": "summarized"}
# Flipped off permanently if the configured model rejects the parameter.
_thinking_supported = True


def _truncate_tool_result(result: str, limit: int) -> str:
    """Cap a tool result, saying plainly that re-reading will not help.

    A bare "…[truncated]" invites the model to try again for the rest; it then
    gets the identical fragment and loops.
    """
    if len(result) <= limit:
        return result
    return (
        result[:limit]
        + f"\n…[TRUNCATED: showing the first {limit:,} of {len(result):,} characters. "
        "Reading this file again returns exactly this same fragment — do not "
        "retry. Work from what is shown above.]"
    )


# ---------------------------------------------------------------------------
# Anthropic prompt caching
#
# Render order is tools → system → messages, and caching is a prefix match: any
# byte change invalidates everything after it. Every `system` string reaching
# _loop is a module-level constant or _build_system(template_key), so the whole
# prefix is already stable within a run — the volatile parts (project map, git
# log, change request) live in the user message, after the breakpoint.
#
# Four breakpoints are allowed per request and all four are used: one on the last
# tool, one on the system block, and a rolling pair on the conversation so the
# growing tool-result history is re-read from cache instead of re-billed in full
# on every iteration.
# ---------------------------------------------------------------------------

def _cached_system(system: str) -> list[dict[str, Any]]:
    """The system prompt as one cacheable content block."""
    return [{"type": "text", "text": system, "cache_control": dict(_CACHE_CONTROL)}]


def _cached_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool definitions with a cache breakpoint on the last one.

    Tools render before the system prompt, so this breakpoint keeps the tool
    array cached across phases even though each phase has its own system prompt.
    """
    if not tools:
        return list(tools)
    cached = [dict(t) for t in tools]
    cached[-1]["cache_control"] = dict(_CACHE_CONTROL)
    return cached


# Built once so the bytes are identical on every request — rebuilding per call
# would be equivalent here, but a single object makes accidental drift impossible.
_CACHED_TOOLS = _cached_tools(TOOLS)

# Cached per tool set, keyed by the tool names it contains. Phases advertise
# different tool lists (only the reviewing phases get report_findings), and the
# tools array renders before everything else, so each variant has to be one
# byte-identical object or its cache entry is rewritten on every request.
_CACHED_TOOL_SETS: dict[tuple[str, ...], list[dict[str, Any]]] = {}


def _cached_tools_for(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return the cache-annotated form of `tools`, stable across calls."""
    resolved = tools if tools is not None else TOOLS
    key = tuple(t["name"] for t in resolved)
    if key not in _CACHED_TOOL_SETS:
        _CACHED_TOOL_SETS[key] = _cached_tools(resolved)
    return _CACHED_TOOL_SETS[key]


def _block_type(block: Any) -> str | None:
    """Content-block type, for blocks that may be dicts or SDK objects."""
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _markable_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last content block of each message we built ourselves.

    Only blocks we construct (tool_result dicts) can carry a marker; assistant
    turns hold SDK objects and are left untouched.
    """
    return [
        msg["content"][-1]
        for msg in messages
        if isinstance(msg.get("content"), list)
        and msg["content"]
        and isinstance(msg["content"][-1], dict)
    ]


def _mark_message_breakpoints(
    messages: list[dict[str, Any]], anchor_end: int, trimming: bool,
) -> None:
    """Place the conversation's cache breakpoints, differently once trimming starts.

    While the whole history still fits, marking the two newest turns caches the
    entire conversation: each request re-reads everything up to the previous turn.
    Measured hit rate in that regime is ~50% of message breakpoints, covering all
    the history.

    Once the window starts sliding, that stops working completely. The window
    drops one pair per turn, so the message prefix differs on every single
    request and no message breakpoint can ever match a stored one — measured at
    0/30. Left alone those markers are worse than useless: each one still pays
    the 1.25x cache-write premium and never earns a read back.

    So above the trim point the only breakpoint that can pay is one at the end of
    the immutable anchor, which by construction never changes.
    """
    for block in _markable_blocks(messages):
        block.pop("cache_control", None)

    if trimming:
        # Anchor only — the tail is unstable by definition.
        anchor_blocks = _markable_blocks(messages[:anchor_end])
        keep = anchor_blocks[-1:]
    else:
        keep = _markable_blocks(messages)[-2:]

    for block in keep:
        block["cache_control"] = dict(_CACHE_CONTROL)


def _cache_stats(usage: Any) -> tuple[int, int, int]:
    """(fresh, cache_creation, cache_read) input tokens for one turn."""
    return (
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
    )


def _usage_tokens(usage: Any) -> int:
    """Total prompt+completion tokens for one turn, including cached input.

    `input_tokens` counts only the uncached remainder once caching is on, so
    summing it alone would make builds look dramatically cheaper than they are.
    Adding the cached halves back keeps this number comparable with the
    pre-caching figures the other backends still report.
    """
    return (
        (getattr(usage, "input_tokens", 0) or 0)
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        + (getattr(usage, "output_tokens", 0) or 0)
    )


def _log_cache_usage(usage: Any) -> None:
    """Record the cache split so the saving is measurable rather than assumed.

    This goes to the application log, not log_fn: the build feed is a user-facing
    narration of the work, and per-iteration cache accounting is operator
    telemetry.
    """
    logger.info(
        "anthropic usage: input=%d cache_creation=%d cache_read=%d output=%d",
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Template-specific directory / component scaffolding guidance
# ---------------------------------------------------------------------------

_FLUTTER_STRUCTURE = """
ARCHITECTURE: Clean Architecture with Feature-first organisation.

DATABASE RULE (if a database is connected — Supabase/Neon/Firebase): the SDK is
initialized and queried ONLY inside data/datasources/{feature}_remote_datasource.dart.
data/repositories/{feature}_repository_impl.dart calls only the datasource — never
the SDK directly. Business/validation logic lives in domain/usecases/, not in the
datasource or repository. presentation/bloc never imports a datasource or the SDK.

FORGEFY CLIENT SDK — use it when the connected database is Supabase or Neon:
Use the first-party `forgefy_client` package for ALL database reads/writes and
auth. Do NOT hand-roll api_client.dart for the DB and do NOT add supabase_flutter
— forgefy_client already ships the retry / error / session layer.
  • pubspec.yaml → add  forgefy_client: ^0.1.0
  • lib/core/network/forgefy_client.dart — build ONE ForgefyClient from env
    (SUPABASE_URL + SUPABASE_ANON_KEY; or ForgefyConfig(provider:
    ForgefyProvider.neon, url: NEON_DATA_API_URL, anonKey: ...)) with a
    SessionStore backed by shared_preferences so logins survive restarts.
    Register it as a lazy singleton in lib/core/injection.dart and inject it
    into the datasources. This file REPLACES api_client.dart for Supabase/Neon.
  • {feature}_remote_datasource.dart calls client.from('table')… ; auth
    datasources call client.auth.signInWithPassword(...) / signUp(...) — never
    raw http or a provider SDK. Read the result, THEN cast (precedence matters):
        final result = await client
            .from('todos').select().eq('done', false)
            .order('created_at', ascending: false);
        final rows = result as List;
  • On startup (main.dart) call client.auth.restoreSession() before runApp.
  FIREBASE apps keep using the firebase_* SDKs — forgefy_client does not cover
  Firebase yet. A plain external REST API (not the DB) still uses api_client.dart.

COMPONENT LIBRARY — shadcn_ui (the house control kit for Flutter):
Use shadcn_ui's Shad* widgets for interactive controls in place of raw Material
widgets, hand-rolled ones, AND the design system's AppButton/AppTextField — this
SUPERSEDES the "use AppTextField/AppButton" note for CONTROL widgets. Widgets:
ShadButton, ShadInput / ShadInputFormField, ShadCard, ShadDialog (showShadDialog)
/ ShadSheet, ShadSelect, ShadCheckbox, ShadSwitch, ShadTabs, ShadBadge,
ShadTooltip, ShadForm (+ validation).
  • pubspec.yaml → add  shadcn_ui: ^0.55.0
  • ROOT (lib/main.dart): a ShadTheme must be in the tree for Shad* to render.
    This template uses MaterialApp.router + GoRouter, so use ShadApp.router(
    routerConfig: appRouter, …) — it provides BOTH the Material theme and the
    ShadTheme — and keep the existing AppTheme via its materialThemeBuilder. Build
    the ShadThemeData colorScheme FROM AppColors (light + dark) so the two match; do
    NOT introduce a second, clashing palette.
  • Icons: LucideIcons (shadcn_ui bundles lucide).
COEXISTENCE — additive, not a rewrite:
  • TOKENS stay the design system's — colors via AppColors, text via AppTextStyles,
    spacing from the theme. Shad* controls are themed FROM those tokens, never
    hardcoded.
  • LAYOUT stays forgefy_ui / sliver-first (VStack/Grid/SliverScreen/…) when the
    Forgefy UI package is enabled; otherwise standard Flutter layout widgets.

EXACT FOLDER STRUCTURE — call create_directory for every path below before writing any files:

  lib/core/error/
  lib/core/network/
  lib/core/usecases/
  lib/core/utils/
  lib/core/theme/

  For EACH feature extracted from the blueprint, create ALL of these:
  lib/features/{feature}/data/datasources/
  lib/features/{feature}/data/models/
  lib/features/{feature}/data/repositories/
  lib/features/{feature}/domain/entities/
  lib/features/{feature}/domain/repositories/
  lib/features/{feature}/domain/usecases/
  lib/features/{feature}/presentation/bloc/
  lib/features/{feature}/presentation/pages/
  lib/features/{feature}/presentation/widgets/

  NOTE: if the feature is "auth" (login / register / session), create it ONLY if
  auth decision is YES. If auth decision is NO, do not create the auth feature folder
  or any login/register pages — skip it entirely.

  assets/images/
  assets/videos/   (if video assets are needed)

CANONICAL FILE NAMES — use these exact names, no variations:

  lib/core/error/exceptions.dart        — AppException subclasses
  lib/core/error/failures.dart          — Failure sealed class / subclasses
  lib/core/network/api_client.dart      — Dio/http base client with interceptors
                                          (external REST API only; for a Supabase/Neon
                                          DB use lib/core/network/forgefy_client.dart)
  lib/core/network/network_info.dart    — connectivity check
  lib/core/usecases/usecase.dart        — abstract UseCase<Type, Params> interface
  lib/core/utils/constants.dart         — API base URL, timeout durations, string keys
  lib/core/theme/app_theme.dart         — ThemeData (light + dark), palette, typography

  lib/features/{feature}/data/datasources/{feature}_remote_datasource.dart
  lib/features/{feature}/data/datasources/{feature}_local_datasource.dart
  lib/features/{feature}/data/models/{entity}_model.dart
  lib/features/{feature}/data/repositories/{feature}_repository_impl.dart
  lib/features/{feature}/domain/entities/{entity}.dart
  lib/features/{feature}/domain/repositories/{feature}_repository.dart
  lib/features/{feature}/domain/usecases/{action}_usecase.dart
  lib/features/{feature}/presentation/bloc/{feature}_bloc.dart
  lib/features/{feature}/presentation/bloc/{feature}_event.dart
  lib/features/{feature}/presentation/bloc/{feature}_state.dart
  lib/features/{feature}/presentation/pages/{feature}_page.dart
  lib/features/{feature}/presentation/widgets/{feature}_form.dart  (or _card, _tile, etc.)

ROOT FILES (these already EXIST in the template — extend them, do NOT create lib/app.dart):
  lib/core/injection.dart       — GetIt registering all blocs, repos, usecases, datasources
  lib/router/app_router.dart    — GoRouter route definitions (add new routes here)
  lib/main.dart                 — runApp + MaterialApp.router(routerConfig: appRouter,
                                  theme: AppTheme.light/dark), wrapped in the BlocProviders,
                                  DI initialised before runApp. There is NO lib/app.dart —
                                  the root MaterialApp lives in main.dart.

pubspec.yaml — add: flutter_bloc, equatable, get_it, dartz, dio, shared_preferences,
               connectivity_plus, shadcn_ui (^0.55.0 — the control kit),
               forgefy_client (^0.1.0, when the DB is Supabase/Neon),
               and any feature-specific packages (firebase_*, etc.)

BUILD ORDER (strictly follow):
  1. core/ files first (error, network, utils, theme)
  2. domain/ layer for every feature (entities → repositories → usecases)
  3. data/ layer for every feature (models → datasources → repository_impl)
  4. presentation/widgets/ for every feature (reusable sub-widgets)
  5. presentation/bloc/ for every feature
  6. presentation/pages/ for every feature
  7. lib/core/injection.dart (wire everything together)
  8. lib/router/app_router.dart (routes) + lib/main.dart (root MaterialApp.router)
  9. Generate all image/video assets, declare in pubspec.yaml
  10. pubspec.yaml — finalize all dependencies
"""

_NEXT_STRUCTURE = """
ARCHITECTURE: Next.js 14 App Router — Server-side API Routes + Client Components.

RULE: Any operation that touches a database, reads secrets, or must be protected
goes in app/api/**  (Route Handlers running on the server).
Client pages/components call these API routes via fetch — they never import server
modules directly.

DATABASE RULE (if a database is connected — Supabase / Neon / Firebase): ALL reads
and writes go through lib/services/{entity}.ts, never inline queries scattered across
route handlers or pages. Route handlers call lib/services/* functions; they never
call the database client (lib/db.ts) or a provider SDK directly. This is true even
for public-safe values like a Supabase anon key or Firebase client config — the
database client/SDK is initialized and queried ONLY inside lib/db.ts and
lib/services/*.ts, and lib/services/*.ts is imported ONLY by app/api/** route
handlers, NEVER by a "use client" component.

FORGEFY CLIENT SDK — use it when the connected database is Supabase or Neon:
Use `@forgefy/client` for ALL database reads/writes and auth. Do NOT add
@supabase/supabase-js and do NOT hand-roll a fetch wrapper for the DB — the
package ships the retry / error / session layer and runs on the edge runtime.
  • package.json → add  "@forgefy/client": "^0.1.0"
  • lib/db.ts — export a factory that builds a ForgefyClient from
    NEXT_PUBLIC_SUPABASE_URL + NEXT_PUBLIC_SUPABASE_ANON_KEY (or provider: 'neon'
    with NEXT_PUBLIC_NEON_DATA_API_URL). In a route handler, forward the caller's
    JWT so queries run as that user (RLS applies):
        const token = req.headers.get('authorization')?.replace('Bearer ', '');
        return new ForgefyClient({ url, anonKey, accessToken: token });
  • lib/services/{entity}.ts calls db.from<T>('table').select()… / .insert(…),
    and auth services call db.auth.signInWithPassword(...) — imported ONLY by
    app/api/** route handlers, never a "use client" component.
  FIREBASE apps keep using the firebase SDK — forgefy_client does not cover
  Firebase yet.

COMPONENT LIBRARY — components/ui (rich custom kit + shadcn/ui primitives):
The template ships a rich custom component kit AND real shadcn/Radix primitives
(new-york style). Reuse them — do NOT hand-roll buttons, inputs, cards, dialogs,
menus, etc.
  • Rich custom controls — import from the barrel "@/components/ui": Button
    (variants + isLoading + leading/trailingIcon), Card (+ CardHeader/CardTitle/…),
    TextField (label/hint/error + password toggle), Chip, EmptyState, Modal, Avatar,
    Skeleton, SignatureWidget. PREFER these for buttons, text inputs, and cards —
    they are richer than the shadcn defaults.
  • shadcn/Radix primitives — import from "@/components/ui/<name>": Dialog, Select,
    DropdownMenu, Tabs, Tooltip, Popover, Sheet, Separator, Label. Use these for
    overlays, menus, selects, tabs, tooltips.
  • Class-merge helper: import { cn } from "@/lib/utils". Icons: "lucide-react".
  • Need another shadcn component (accordion, command, checkbox, …)? Write it from
    the official shadcn (new-york) source at components/ui/<name>.tsx (you cannot run
    the shadcn CLI) and add its @radix-ui/react-<primitive> dep to package.json.
  • Colors/spacing use the design-system Tailwind tokens (bg-primary, bg-surface,
    text-on-surface, text-muted-foreground, border-border) — never hardcode hex.
    The shadcn primitives already map to these tokens.
Components live in components/ui; lay them out with Tailwind flex/grid (or the
@forgefy/ui/web layout primitives when enabled).

EXACT FOLDER STRUCTURE — call create_directory for EVERY path before writing files:

  ── Server-side API routes (run on the server, never sent to the browser) ──
  app/api/auth/login/           [AUTH ONLY]
  app/api/auth/register/        [AUTH ONLY]
  app/api/auth/logout/          [AUTH ONLY]
  app/api/auth/me/              [AUTH ONLY]
  For each feature from the blueprint:
    app/api/{feature}/          — collection: GET (list) + POST (create)
    app/api/{feature}/[id]/     — item:       GET (single) + PUT (update) + DELETE

  ── Client-facing pages (App Router groups) ──
  app/(auth)/login/             [AUTH ONLY]
  app/(auth)/register/          [AUTH ONLY]
  app/(app)/layout.tsx          — authenticated shell, checks session  [AUTH ONLY if auth required; otherwise use a plain layout]
  For each feature:
    app/(app)/{feature}/        — feature list/dashboard page
    app/(app)/{feature}/[id]/   — feature detail page (if needed)

  ── Shared UI ──
  components/ui/                — Button, Input, Card, Modal, Spinner, Badge, Avatar,
                                   Select, Checkbox, Textarea, Toast, Dialog, Tooltip
  components/layout/            — Header, Footer, Sidebar, PageWrapper, MobileNav, Breadcrumb
  For each feature:
    components/{feature}/       — feature-specific reusable components

  ── Server utilities (imported only by app/api/**) ──
  lib/db.ts                     — database client (ForgefyClient factory for Supabase/Neon —
                                   see FORGEFY CLIENT SDK above; else Prisma / mongoose / firebase)
  For each entity/feature that needs persistence:
    lib/services/{entity}.ts    — the ONLY place with queries for this entity: getX/listX,
                                   createX, updateX, deleteX — imported only by app/api/**
  lib/auth.ts                   — session helpers, JWT sign/verify, cookie utilities  [AUTH ONLY]
  lib/validations.ts            — Zod schemas for validating request bodies

  ── Client utilities (safe to import in client components) ──
  lib/api.ts                    — typed fetch wrapper that calls /api/* routes
  lib/utils.ts                  — shared pure helpers (formatDate, cn, etc.)

  hooks/                        — useAuth.ts [AUTH ONLY], use{Feature}.ts — SWR / React Query hooks
  types/                        — index.ts with all shared TypeScript interfaces
  middleware.ts                 — Next.js edge middleware (protects /app/* routes)  [AUTH ONLY]
  public/images/                — AI-generated assets

SERVICES LAYER PATTERN — use this shape for every lib/services/{entity}.ts:
  import {{ db }} from '@/lib/db'
  import type {{ {Entity} }} from '@/types'

  export async function list{Entity}s(/* filters */): Promise<{Entity}[]> {{
    return db...
  }}
  export async function create{Entity}(data: ...): Promise<{Entity}> {{
    return db...
  }}
  // update{Entity}, delete{Entity}, get{Entity}ById follow the same shape

API ROUTE PATTERN — use this shape for every route.ts (call lib/services/*, never db directly):
  import {{ NextRequest, NextResponse }} from 'next/server'
  import {{ z }} from 'zod'
  import {{ list{Entity}s, create{Entity} }} from '@/lib/services/{entity}'
  import {{ getSession }} from '@/lib/auth'

  const Schema = z.object({{ ... }})

  export async function GET(req: NextRequest) {{
    const session = await getSession(req)
    if (!session) return NextResponse.json({{ error: 'Unauthorized' }}, {{ status: 401 }})
    const data = await list{Entity}s()
    return NextResponse.json(data)
  }}

  export async function POST(req: NextRequest) {{
    const session = await getSession(req)
    if (!session) return NextResponse.json({{ error: 'Unauthorized' }}, {{ status: 401 }})
    const body = Schema.parse(await req.json())
    const result = await create{Entity}(body)
    return NextResponse.json(result, {{ status: 201 }})
  }}

BUILD ORDER (strictly follow; skip [AUTH ONLY] steps if auth decision is NO):
  1.  types/index.ts                        — all shared TypeScript interfaces
  2.  lib/db.ts                             — database client
  3.  lib/services/{entity}.ts              — one per entity/feature that needs persistence
  4.  lib/auth.ts                           — session / JWT helpers              [AUTH ONLY]
  5.  lib/validations.ts                    — Zod schemas (one per resource)
  6.  lib/api.ts                            — client-side fetch wrapper
  7.  lib/utils.ts                          — shared helpers
  8.  app/api/auth/login/route.ts           — POST: validate → hash → session    [AUTH ONLY]
  9.  app/api/auth/register/route.ts        — POST: validate → hash → user       [AUTH ONLY]
  10. app/api/auth/logout/route.ts          — POST: clear session cookie          [AUTH ONLY]
  11. app/api/auth/me/route.ts              — GET: return session user             [AUTH ONLY]
  12. For each feature: app/api/{f}/route.ts and app/api/{f}/[id]/route.ts (call lib/services/*)
  13. middleware.ts                         — protect (app)/* group               [AUTH ONLY]
  14. components/ui/*.tsx                  — ALL base components before any page
  15. components/layout/*.tsx              — layout components
  16. hooks/useAuth.ts                     — auth hook                            [AUTH ONLY]
      hooks/use{Feature}.ts               — SWR/RQ hooks calling lib/api.ts
  17. app/layout.tsx                       — root layout with providers
  18. app/(auth)/login/page.tsx            — login form, calls /api/auth/login    [AUTH ONLY]
  19. app/(auth)/register/page.tsx         — register form, calls /api/auth/register [AUTH ONLY]
  20. app/(app)/layout.tsx                 — authenticated shell                  [AUTH ONLY → otherwise use plain app/layout.tsx]
  21. For each feature: app/(app)/{f}/page.tsx using the components and hooks
  22. Generate all image/video assets, reference in pages
  23. tailwind.config.ts / globals.css     — theme tokens
  24. package.json                         — finalize dependencies (zod, swr, etc.)

TYPESCRIPT RULES — follow exactly to avoid build errors:

  PROPS: always define an interface, never use React.FC<>
    ✓  interface ButtonProps {{ label: string; onClick: () => void }}
       function Button({{ label, onClick }}: ButtonProps) {{ ... }}
    ✗  const Button: React.FC<{{ label: string }}> = ({{ label }}) => {{ ... }}

  ASYNC ROUTE HANDLERS: always type the return explicitly
    ✓  export async function GET(req: NextRequest): Promise<NextResponse> {{ ... }}

  DYNAMIC ROUTE PARAMS: Next.js 14 passes params as a Promise — await them
    ✓  export default async function Page({{ params }}: {{ params: Promise<{{ id: string }}> }}) {{
         const {{ id }} = await params
       }}
    ✗  export default function Page({{ params }}: {{ params: {{ id: string }} }}) {{ ... }}

  STATE: always provide the generic type
    ✓  const [items, setItems] = useState<Item[]>([])
    ✗  const [items, setItems] = useState([])

  UNKNOWN VALUES: use 'unknown' then narrow, or 'any' if narrowing is impractical
    ✓  catch (err: unknown) {{ const msg = err instanceof Error ? err.message : String(err) }}
    ✗  catch (err) {{ console.error(err.message) }}

  IMPORTS: import types with 'import type' to avoid runtime issues
    ✓  import type {{ User }} from '@/types'

  next.config.js — always include these two flags so the build never fails on type errors:
    typescript: {{ ignoreBuildErrors: true }}
    eslint: {{ ignoreDuringBuilds: true }}
"""

_RN_STRUCTURE = """
ARCHITECTURE: Expo Router (file-based routing) + feature modules + Zustand + React Query.
Routes live in app/ (expo-router — the filename IS the route). Shared code in src/core/.
Feature logic in src/features/{feature}/. Client state via Zustand; server data via
@tanstack/react-query. This template is NOT React Navigation and NOT Redux — do not add
@react-navigation/* or @reduxjs/toolkit.

FORGEFY CLIENT SDK — use it when the connected database is Supabase or Neon:
Use `@forgefy/client` for ALL database reads/writes and auth. Do NOT add
@supabase/supabase-js — the package ships the retry / error / session layer.
  • package.json → add  "@forgefy/client": "^0.1.0"
  • src/services/forgefyClient.ts — build ONE ForgefyClient from
    EXPO_PUBLIC_SUPABASE_URL + EXPO_PUBLIC_SUPABASE_ANON_KEY (or provider: 'neon'
    with EXPO_PUBLIC_NEON_DATA_API_URL), with a persistent session store backed by
    expo-secure-store or react-native-mmkv (wrap it to match persistentSessionStore's
    getItem/setItem/removeItem — this template has NO AsyncStorage). Call
    forgefy.auth.restoreSession() on app start (in app/_layout.tsx).
  • src/services/db/{entity}.ts calls forgefy.from('table')… and auth calls
    forgefy.auth.signInWithPassword(...) — never a provider SDK from a route,
    component, store, or hook.
  FIREBASE apps keep using the firebase SDK — forgefy_client does not cover
  Firebase yet. src/services/httpClient.ts stays for a plain external REST API.

COMPONENT LIBRARY — react-native-reusables (the shadcn kit for React Native):
react-native-reusables (RNR) is the RN shadcn port — NativeWind (Tailwind
classNames) + @rn-primitives/* + class-variance-authority + cn(). Components are
copied in under components/ui/.
  • USE IT ONLY WHEN the template already has NativeWind configured — a global.css
    with the theme CSS variables, a tailwind.config.js RNR preset, the NativeWind
    babel/metro setup, and cn() in lib/utils. Then compose controls from
    components/ui/* (Button, Input, Card, Dialog, Select, Text, Badge, …) instead of
    raw react-native controls or hand-rolled ones. Icons: lucide-react-native.
  • If a needed RNR component isn't in components/ui/, WRITE it from the official RNR
    source (you cannot run the RNR CLI) and add its deps to package.json: the matching
    @rn-primitives/<name>, plus class-variance-authority, clsx, tailwind-merge (and
    nativewind, tailwindcss, tailwindcss-animate if not already present).
  • DO NOT introduce NativeWind mid-build if the template does not already use it —
    the babel/metro/global.css changes break the build if imperfect. In that case use
    the pre-built src/core/components/ instead.
COEXISTENCE: RNR = controls; design TOKENS stay src/core/theme's; LAYOUT stays
@forgefy/ui/native (VStack/Grid/List/…) when enabled. NativeWind className and RN
style props coexist, so forgefy_ui (style-based) works alongside RNR (className).

EXACT FOLDER STRUCTURE — call create_directory for every path before writing files:

  ── Routes (expo-router; the filename IS the route. _layout.tsx files already exist) ──
  app/_layout.tsx               — root Stack + global providers (EXISTS — extend, don't recreate)
  app/(tabs)/_layout.tsx        — bottom Tabs navigator (EXISTS)
  app/(tabs)/{name}.tsx         — a tab screen (index.tsx = Home)
  For each feature that needs its own route(s):
    app/{feature}/_layout.tsx   — nested Stack (only if the feature has sub-screens)
    app/{feature}/index.tsx     — feature home/list route
    app/{feature}/[id].tsx      — feature detail route (dynamic segment)
  [AUTH ONLY] app/(auth)/_layout.tsx, app/(auth)/login.tsx, app/(auth)/register.tsx

  ── Feature modules (src/features/{feature}/) ──
  For each feature from the blueprint:
    src/features/{feature}/components/   — feature UI (built on src/core + src/components/ui)
    src/features/{feature}/hooks/        — use{Feature}.ts (React Query + Zustand)
    src/features/{feature}/store.ts      — Zustand store (only if the feature has client state)
    src/features/{feature}/api.ts        — React Query queries/mutations
    src/features/{feature}/types.ts      — TypeScript interfaces for this feature

  NOTE: if a feature is "auth" (login / register / session), create it ONLY if auth
  decision is YES. If auth decision is NO, do not create the auth routes, store, or API.

  ── Shared (already scaffolded — do NOT recreate) ──
  src/core/components/          — custom control kit (AppButton, AppCard, AppTextField, …)
  src/components/ui/            — react-native-reusables components (when NativeWind is set up)
  src/core/theme/               — ThemeProvider, tokens.ts, useTheme (useAppTheme())
  src/core/utils/               — formatters, haptics, helpers
  src/core/types/               — shared types
  For each entity that needs persistence:
    src/services/db/{entity}.ts — the ONLY place that reads/writes the connected DB —
                                   never call the DB/SDK from a route, component, hook, or store
  assets/images/                — AI-generated assets

CANONICAL NAMES / RULES:
  • Routes are files under app/ — filename = URL segment. Groups use (parens), layouts
    are _layout.tsx, dynamic segments are [id].tsx. Add a tab by creating the file AND a
    <Tabs.Screen name="..."/> in app/(tabs)/_layout.tsx.
  • Navigation: import { router, useRouter, Link } from 'expo-router' — router.push('/x/1').
    Do NOT import @react-navigation directly.
  • Client state: Zustand — create() a store in src/features/{feature}/store.ts (or src/core
    for global). Do NOT use Redux / @reduxjs/toolkit / configureStore / createSlice.
  • Server data: @tanstack/react-query (useQuery / useMutation) in feature api.ts / hooks.
  • Theme: read colors/spacing via useAppTheme() from src/core/theme — never hardcode hex.
  • Persistent storage: react-native-mmkv (fast KV) or expo-secure-store (secrets).
    Do NOT use @react-native-async-storage/async-storage.
  • Path aliases (tsconfig): @core/* → src/core/*, @features/* → src/features/*,
    ~/* → src/*  (react-native-reusables), @assets/* → assets/*.

ROOT (already exists — EXTEND, do NOT recreate):
  app/_layout.tsx — root Stack wrapped in QueryClientProvider + ThemeProvider +
    GestureHandlerRootView + BottomSheetModalProvider. Add new global providers here.

BUILD ORDER (strictly follow; skip [AUTH ONLY] steps if auth decision is NO):
  1. src/features/{feature}/types.ts for every feature
  2. src/services/db/{entity}.ts for every entity that needs persistence (if a DB is connected)
  3. src/features/{feature}/api.ts (React Query) + store.ts (Zustand) per feature
  4. src/features/{feature}/hooks/use{Feature}.ts
  5. src/features/{feature}/components/ (feature UI using core + ui components)
  6. app/ route files per feature (thin — render feature components); add <Tabs.Screen> entries
  7. [AUTH ONLY] app/(auth)/* routes + session handling
  8. Generate image assets, reference them in the routes
  9. app.json + package.json — finalize app name and dependencies
"""

_STRUCTURE_MAP = {
    "flutter": _FLUTTER_STRUCTURE,
    "next": _NEXT_STRUCTURE,
    "react_native": _RN_STRUCTURE,
}

# Compact structure rules injected into update-agent user messages.
# These are shorter than the full build-agent structures — focused on
# "where does a new file go?" rather than "how do I scaffold the whole app?"
_UPDATE_STRUCTURE_RULES: dict[str, str] = {
    "flutter": """\
FLUTTER FOLDER STRUCTURE — every file you create MUST follow this layout:

  lib/core/error/exceptions.dart          lib/core/error/failures.dart
  lib/core/network/api_client.dart        lib/core/network/network_info.dart
  lib/core/usecases/usecase.dart
  lib/core/utils/constants.dart
  lib/core/theme/app_theme.dart

  For each feature → ALL sub-folders are mandatory:
  lib/features/{feature}/data/datasources/{feature}_remote_datasource.dart
  lib/features/{feature}/data/datasources/{feature}_local_datasource.dart
  lib/features/{feature}/data/models/{entity}_model.dart
  lib/features/{feature}/data/repositories/{feature}_repository_impl.dart
  lib/features/{feature}/domain/entities/{entity}.dart
  lib/features/{feature}/domain/repositories/{feature}_repository.dart
  lib/features/{feature}/domain/usecases/{action}_usecase.dart
  lib/features/{feature}/presentation/bloc/{feature}_bloc.dart
  lib/features/{feature}/presentation/bloc/{feature}_event.dart
  lib/features/{feature}/presentation/bloc/{feature}_state.dart
  lib/features/{feature}/presentation/pages/{feature}_page.dart
  lib/features/{feature}/presentation/widgets/{feature}_*.dart

  Root (already exist — extend, do NOT create lib/app.dart):
  lib/core/injection.dart   lib/router/app_router.dart (GoRouter)   lib/main.dart (root MaterialApp.router)

RULES:
  • Pages go in presentation/pages/ ONLY. Never at lib/ root or lib/screens/.
  • Widgets go in presentation/widgets/ ONLY.
  • Data models go in data/models/ ONLY.
  • If a database is connected (Supabase/Neon/Firebase), the SDK is initialized and
    queried ONLY inside data/datasources/{feature}_remote_datasource.dart — the
    repository_impl calls only the datasource, business/validation logic lives in
    domain/usecases/, and presentation/bloc never touches the datasource or SDK directly.
    For Supabase/Neon that SDK is `forgefy_client`. Build ONE ForgefyClient in
    lib/core/network/forgefy_client.dart from SUPABASE_URL + SUPABASE_ANON_KEY (or
    ForgefyProvider.neon + NEON_DATA_API_URL) with a shared_preferences SessionStore,
    register it in lib/core/injection.dart, and add forgefy_client: ^0.1.0 to
    pubspec.yaml. Datasources call client.from('table').select()… ; auth calls
    client.auth.signInWithPassword(...). Read the result THEN cast (precedence):
        final r = await client.from('t').select().eq('done', false); final rows = r as List;
    Firebase apps keep the firebase_* SDKs.
  • COMPONENTS: use shadcn_ui's Shad* widgets (ShadButton/ShadInput/ShadCard/
    ShadDialog/ShadSelect/…) for controls instead of raw Material or AppButton/
    AppTextField. Add shadcn_ui: ^0.55.0; a ShadTheme must be at the root — this
    template uses MaterialApp.router, so ShadApp.router in lib/main.dart, colorScheme
    built from AppColors. Tokens still come from AppColors/AppTextStyles.
  • If you add a new feature, create ALL sub-folders listed above.
  • If you add a new screen, add its route to lib/router/app_router.dart (GoRouter).
""",
    "react_native": """\
REACT NATIVE STRUCTURE (Expo Router + Zustand + React Query) — every file MUST follow this:

  Routes live in app/ (expo-router — filename = route; _layout.tsx already exist):
  app/_layout.tsx                   ← root Stack + providers (EXISTS — extend, don't recreate)
  app/(tabs)/_layout.tsx            ← bottom Tabs (add <Tabs.Screen> to register a tab)
  app/(tabs)/{name}.tsx             ← a tab screen (index.tsx = Home)
  app/{feature}/index.tsx  app/{feature}/[id].tsx   ← feature routes (dynamic = [id].tsx)

  Feature logic in src/features/{feature}/:
  src/features/{feature}/components/    src/features/{feature}/hooks/use{Feature}.ts
  src/features/{feature}/store.ts (Zustand)   src/features/{feature}/api.ts (React Query)
  src/features/{feature}/types.ts

  Shared (already scaffolded — do NOT recreate):
  src/core/components/ (AppButton/AppCard/… custom kit)   src/components/ui/ (RNR, if NativeWind)
  src/core/theme/ (useAppTheme())   src/core/utils/   src/core/types/
  src/services/db/{entity}.ts   ← ONE per entity — the ONLY place that reads/writes the DB

RULES:
  • Routes go in app/ ONLY (expo-router). Add a screen by creating its file; register a tab
    with <Tabs.Screen name="..."/> in app/(tabs)/_layout.tsx. Navigate via
    import { router } from 'expo-router'. Do NOT use @react-navigation directly.
  • This template is Zustand + React Query — NOT Redux. No store.ts/rootReducer/createSlice/
    useAppDispatch. Client state → Zustand create() in src/features/{feature}/store.ts;
    server data → React Query in api.ts/hooks. Theme via useAppTheme(); storage via
    react-native-mmkv / expo-secure-store (NOT AsyncStorage).
  • If a database is connected, ALL reads/writes go through src/services/db/{entity}.ts
    — routes/components/hooks/stores never call Supabase/Neon/Firebase SDKs directly.
    For Supabase/Neon that SDK is `@forgefy/client`. Build ONE ForgefyClient in
    src/services/forgefyClient.ts from EXPO_PUBLIC_SUPABASE_URL + EXPO_PUBLIC_SUPABASE_ANON_KEY
    (or provider:'neon' + EXPO_PUBLIC_NEON_DATA_API_URL) with a session store backed by
    expo-secure-store/mmkv; call forgefy.auth.restoreSession() in app/_layout.tsx. Add
    "@forgefy/client": "^0.1.0" to package.json. Firebase apps keep the firebase SDK.
  • COMPONENTS: if the template uses NativeWind, use react-native-reusables from
    src/components/ui/* (Button/Input/Card/Dialog/Select/…) for controls instead of raw
    RN or hand-rolled ones; write a missing RNR component from source and add its
    @rn-primitives/* dep (+ cva/clsx/tailwind-merge). If the template does NOT use
    NativeWind, do NOT add it mid-build — use src/core/components/. Icons: lucide-react-native.
""",
    "next": """\
NEXT.JS FOLDER STRUCTURE — every file you create MUST follow this layout:

  app/api/{feature}/route.ts          — server-side: GET (list) + POST (create)
  app/api/{feature}/[id]/route.ts     — server-side: GET + PUT + DELETE
  app/(app)/{feature}/page.tsx        — client page for this feature
  components/{feature}/               — feature-specific reusable components
  components/ui/                      — shared primitive components
  components/layout/                  — Header, Sidebar, Footer, etc.
  lib/db.ts   lib/services/{entity}.ts   lib/auth.ts   lib/validations.ts   lib/api.ts   lib/utils.ts
  hooks/use{Feature}.ts
  types/index.ts
  middleware.ts                        — route protection [auth only]

RULES:
  • API logic goes in app/api/ ONLY — never in page.tsx files.
  • If a database is connected (Supabase/Neon/Firebase), ALL reads/writes go through
    lib/services/{entity}.ts — route handlers call these functions, never inline
    queries against lib/db.ts or a provider SDK directly, and never from a
    "use client" component. For Supabase/Neon that SDK is `@forgefy/client`: build the
    ForgefyClient in lib/db.ts from NEXT_PUBLIC_SUPABASE_URL + NEXT_PUBLIC_SUPABASE_ANON_KEY
    (or provider:'neon' + NEXT_PUBLIC_NEON_DATA_API_URL); in a route handler forward the
    caller's JWT — new ForgefyClient({ url, anonKey, accessToken }) — so RLS applies.
    lib/services/{entity}.ts calls db.from<T>('table')… / db.auth. Add
    "@forgefy/client": "^0.1.0" to package.json. Firebase apps keep the firebase SDK.
  • New pages go in app/(app)/{feature}/page.tsx.
  • Add a nav link in components/layout/ when adding a new page.
  • All shared types in types/index.ts — do not scatter them across files.
  • COMPONENTS: use shadcn/ui from @/components/ui/* (new-york style, lucide icons,
    cn() from @/lib/utils) — never hand-roll buttons/inputs/cards/dialogs. If a
    needed component isn't in components/ui/, write it from the shadcn source and add
    its @radix-ui/react-* dep (plus class-variance-authority/clsx/tailwind-merge/
    lucide-react). Colors via Tailwind tokens (bg-background/text-primary), not hex.

TYPESCRIPT — follow exactly to avoid build failures:
  • Props: define an interface, use plain function — NOT React.FC<>
  • Dynamic params (Next.js 14+): params is a Promise — always await it
      async function Page({{ params }}: {{ params: Promise<{{ id: string }}> }}) {{ const {{ id }} = await params }}
  • useState: always provide the generic — useState<Item[]>([])
  • Route handlers: type return as Promise<NextResponse>
  • catch blocks: type as 'unknown', narrow before use
  • next.config.js: always set typescript:{{ignoreBuildErrors:true}} and eslint:{{ignoreDuringBuilds:true}}
""",
}

_DESIGN_MANDATE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN MANDATE — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A design system has been pre-written into this workspace. You MUST treat it
as the single source of truth for all visual decisions.

BEFORE writing any UI file:
1. Read lib/core/theme/app_colors.dart (Flutter)
   OR src/core/theme/tokens.ts (React Native)
   OR lib/design-system/tokens.ts + app/globals.css (Next.js)
2. Read ALL pre-built core/widgets or core/components or components/ui files
3. Use ONLY the design tokens from those files — no hardcoded color hex values,
   no raw pixel values for spacing, no inline font sizes

HARDCODED VALUES ARE FORBIDDEN:
✗  color: Color(0xFF2196F3)        → ✓  color: AppColors.primary
✗  fontSize: 16                    → ✓  style: AppTextStyles.body
✗  padding: EdgeInsets.all(16)     → ✓  padding: EdgeInsets.all(AppSpacing.md)
✗  borderRadius: BorderRadius.circular(8) → ✓  BorderRadius.circular(AppRadius.md)
✗  backgroundColor: '#6366F1'      → ✓  backgroundColor: colors.primary (RN)
✗  color: '#6366F1'                → ✓  color: var(--color-primary) (Next.js)

REUSE COMPONENTS FIRST — do NOT hand-roll standard controls or duplicate:
Standard controls (buttons, inputs, cards, dialogs, selects, tabs, badges, …) come
from the platform component kit, NOT from custom code:
  • Flutter  → shadcn_ui Shad* widgets (ShadButton, ShadInput, ShadCard, …).
  • Next.js  → shadcn/ui in components/ui/ (imported via @/components/ui/*).
  • RN       → react-native-reusables (components/ui/) when NativeWind is set up,
               else the pre-built components in src/core/components/.
For anything app-SPECIFIC (a composite card, an empty state, the signature element),
check lib/core/widgets/ (Flutter) / components/{feature}/ (Next) / feature components
(RN) and reuse/extend rather than duplicate. On Flutter, lib/core/widgets/ holds
composites built ON TOP OF Shad* — not re-implementations of basic controls.

THE SIGNATURE ELEMENT:
The blueprint contains a `signature_element` — one distinctive, domain-specific
design detail. You MUST implement it using the pre-built SignatureWidget
and use it in at least 2 screens where it makes contextual sense.

SCREEN QUALITY FLOOR — every screen must have:
  ⌛ A proper AppBar/header with consistent back navigation
  ⌛ Empty states (not blank screens — use AppEmptyState / EmptyState widget)
  ⌛ Loading states (skeleton loaders, NOT spinners alone)
  ⌛ Error states with actionable messages
  ⌛ At least one micro-interaction (ripple, scale, fade) per primary action
  ⌛ Consistent 16dp/px horizontal page padding
  ⌛ No orphaned text — all text uses a defined text style

LAYOUT RULES:
  ⌛ Lists: always show 3+ items in preview (use realistic placeholder content)
  ⌛ Cards: use consistent elevation and radius from the design system
  ⌛ Forms: group related fields, label everything, show inline validation
  ⌛ Modals/sheets: drag handles on mobile, max-width constraint on web
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

_BUILD_PREAMBLE = """You are the Forgefy Build Agent.
Your task: implement a complete, working application from the blueprint by writing files in the workspace.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEMO SCREEN — REMOVE IT FIRST, WIRE AS YOU GO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The template you cloned ships a demo/placeholder home screen (sample content
that belongs to the template, not to this app). It must NOT survive the build:

1. FIRST TASK — before any feature work: locate the template's demo/placeholder
   screen (a home/welcome/showcase page full of sample content) and DELETE it or
   REPLACE it with this app's real home screen. Update the router/navigator so
   the app opens on real content, not the demo.
2. ENTRY SCREEN FIRST — the launch page is built before everything else. The
   entry screen is whatever opens at launch: app/index.tsx or app/(tabs)/index
   (Expo), app/page.tsx (Next.js), the home widget under lib/ or the initial
   route in MainNavigator (Flutter). Build its REAL layout and content from the
   blueprint first, set it as the initial route, then build every other feature
   in an order that links outward from it. Never start with a leaf screen.
3. INTEGRATE INCREMENTALLY — after finishing EACH page, screen, component or
   feature, IMMEDIATELY register it in the router/navigator (add the route, tab
   entry, or a link from the home screen) BEFORE starting the next one. Never
   batch navigation wiring until the end.
4. WHY: previews are compiled from what is committed. If this run stops early
   (step/token limit), the user sees the last pushed state — it must open on a
   real launch page with the features you actually finished, never the demo
   page. A screen that exists on disk but is not routed is invisible to the
   user.

A build that finishes with the demo screen still showing has failed its first
task, no matter how many feature files it wrote.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUILD PHASES — follow in order
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 0 · Auth Decision  ← do this BEFORE anything else

  Read the blueprint's "features", "entities", and "description" fields.

  AUTH IS REQUIRED if ANY of the following are true:
    • A feature name contains: auth, login, register, sign in, sign up,
      user account, user profile, password, session, onboarding (with accounts)
    • An entity name is: User, Account, Profile, Session, Token, Credential
    • The description mentions: authentication, user accounts, login, sign in,
      protected data, personalisation tied to a user identity

  AUTH IS NOT REQUIRED for: public tools, utilities, dashboards with no user
  model, calculator-type apps, content viewers with no personalisation, public
  APIs, games with no user save state, etc.

  ► Write ONE sentence declaring your decision BEFORE creating any folders:
      "AUTH: YES — [brief reason from blueprint]"
      OR
      "AUTH: NO — [brief reason: public app / no user model / etc.]"

  If AUTH is NO, skip ALL items marked [AUTH ONLY] in the structure below:
    — do NOT create auth API routes, auth feature folder, login/register screens,
      session/JWT code, or auth middleware.
    — treat every build-order step marked [AUTH ONLY] as N/A.
    — build the app as a fully public application with no login wall.

PHASE 0 (auth) — if auth IS needed: the login/register screens MUST use the platform
  control kit — Flutter: shadcn_ui ShadInputFormField + ShadButton (inside a ShadForm);
  Next.js: shadcn/ui Input + Button; RN: react-native-reusables (if NativeWind) else core
  controls. No custom/hand-rolled
  text fields or buttons. Include a branded header using the display_font. Password field
  has a show/hide toggle.

PHASE 1 · Explore
  • list_files on '.' to see the existing template
  • READ the design system files FIRST — before any other action:
      Flutter    → read lib/core/theme/app_colors.dart, lib/core/theme/app_theme.dart,
                   then list lib/core/widgets/ and read each widget file
      RN         → read src/core/theme/tokens.ts, then list src/core/components/
      Next.js    → read app/globals.css, lib/design-system/tokens.ts,
                   then list components/ui/ and read each component
  • Confirm the palette, fonts, and signature_element are understood before proceeding

PHASE 2 · Scaffold directories
  • Call create_directory for every folder listed in the structure below
  • The design system directories (lib/core/theme/, lib/core/widgets/, etc.) already
    exist — do NOT recreate them. Only scaffold feature-specific directories.
  • Before creating each file, write one short sentence narrating what you are doing
    (e.g. "Creating LoginScreen…", "Building AuthService…")
    This sentence appears in the user's live build log.

PHASE 3 · Reusable components / widgets  ← DO THIS BEFORE SCREENS
  • Standard controls come from the component kit — Flutter: shadcn_ui Shad*;
    Next.js: shadcn/ui in components/ui/; RN: react-native-reusables (if NativeWind)
    else pre-built src/core/components/. Do not re-implement them.
  • Only write FEATURE-SPECIFIC components here (composites built ON TOP OF the kit).
    Each must use the kit + design tokens — never redefine base styles.
  • Write at minimum: a feature-specific card, a feature-specific list item, and
    the signature_widget integrated where contextually relevant.

PHASE 4 · Models & services
  • Data models with serialisation (fromJson/toJson, TypeScript interfaces)
  • Enrich models with display helpers: displayTitle, formattedDate, statusLabel getters
  • Service classes: one per domain (auth, API, local storage, etc.)
  • State management wired up

PHASE 5 · Screens / pages
  • Every screen MUST implement the full quality floor (see DESIGN MANDATE above):
      1. AppBar / header
      2. Loading state (skeleton loaders from AppLoading/Skeleton)
      3. Error state with actionable message
      4. Empty state (AppEmptyState/EmptyState with contextual copy and CTA)
      5. Content (list/grid/detail using feature components from Phase 3)
  • Home/dashboard screen: MUST use a visually rich hero section, NOT a plain list.
    Show key metrics, a welcome message, and featured content.

PHASE 6 · Assets
  • Use generate_image for every visual element: backgrounds, hero images,
    onboarding artwork, icons, illustrations — do NOT leave image slots empty
  • Asset image prompts must specify the design system personality and palette.
    Example: "{app_name} — {personality} — hero image, {primary_color} palette,
    {image_treatment} treatment, no text overlays"
  • Use generate_video for splash/onboarding animations where appropriate
  • After generating an asset, immediately reference it in code:
      Flutter    → Image.asset('assets/images/<file>') + declare in pubspec.yaml
      Next.js    → <img src="/images/<file>"> or next/image
      React Native → require('./assets/images/<file>')

PHASE 7 · Configuration
  • Update the app name everywhere (pubspec.yaml, package.json, app.json, Info.plist, etc.)
  • Add all required third-party dependencies
  • Flutter: confirm MaterialApp uses AppTheme.light / AppTheme.dark and ThemeMode.system
  • Verify google_fonts is in pubspec.yaml (Flutter) / design fonts are declared (RN/Next.js)
  • Update bundle ID / application ID

RULES
  • Write real, working code — zero placeholders, zero TODOs
  • Narrate each step (one short sentence) so the user sees live progress
  • After ALL features are implemented, write a clear user-facing summary starting with DONE: that describes what was built — e.g. "DONE: Built a task manager with home screen, add-task form, and local storage. Screens: Home, AddTask, Settings."

"""

_BUILD_SUFFIX = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADDITIONAL REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• If a database is connected for this project, ALL reads and writes must go through
  the services/datasource layer described above for this platform (lib/services/*.ts
  for Next.js, src/services/db/*Service.ts for React Native, data/datasources/* for
  Flutter) — never scatter raw provider SDK/query calls inside pages, screens, or
  widgets/components directly. When that database is Supabase or Neon, that layer uses
  the Forgefy client SDK (forgefy_client for Flutter, @forgefy/client for Next.js and
  React Native) for both data and auth — see the "FORGEFY CLIENT SDK" note in the
  structure section above. Firebase-backed apps keep using the Firebase SDK.
• Implement EVERY feature listed in the blueprint — nothing optional
• Build the ENTRY SCREEN first — the page users see at launch gets its real
  layout and content before any other screen, and stays the initial route
• Remove the template's demo/placeholder home screen (see DEMO SCREEN above) and
  wire each finished page/screen into the router AS YOU GO — an unwired screen
  never reaches a preview, so partial builds must still open on real content
• Handle loading states, empty states, and basic error states in every screen
• Add input validation where the app collects user data
• Style the app consistently using the color/theme constants you define
• Do not leave any generated image/video slot with a placeholder URL
• ALWAYS create THREE environment files:

  1. .env  ← committed to GitHub
     Contains ONLY public-safe variables with placeholder values. Public-safe means:
       — NEXT_PUBLIC_* (bundled into browser JS — cannot be secret by design)
       — Supabase URL + anon key (security enforced by Row Level Security, not the key).
         ALWAYS use these EXACT variable names (a build step may overwrite them with
         a real connected project's values — using any other name means that step
         silently can't find them):
           Next.js        → NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY
           React Native   → EXPO_PUBLIC_SUPABASE_URL, EXPO_PUBLIC_SUPABASE_ANON_KEY
           Flutter        → SUPABASE_URL, SUPABASE_ANON_KEY
         If .env already contains these variables with non-placeholder-looking
         values (not "your-...", not empty), a real database is already connected
         — leave them exactly as they are, do not overwrite with placeholders.
       — Neon Data API URL (public-safe — never the raw Postgres connection
         string, which is a secret and must never appear in any of these files).
         Same rule: ALWAYS use these exact names, and leave real values alone:
           Next.js        → NEXT_PUBLIC_NEON_DATA_API_URL
           React Native   → EXPO_PUBLIC_NEON_DATA_API_URL
           Flutter        → NEON_DATA_API_URL
       — Firebase client config (public-safe — security enforced by Firebase
         Security Rules, not by keeping these values secret): apiKey, authDomain,
         projectId, storageBucket, messagingSenderId, appId.
         Same rule: ALWAYS use these exact names, and leave real values alone:
           Next.js        → NEXT_PUBLIC_FIREBASE_API_KEY, NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
                             NEXT_PUBLIC_FIREBASE_PROJECT_ID, NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
                             NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID, NEXT_PUBLIC_FIREBASE_APP_ID
           React Native   → EXPO_PUBLIC_FIREBASE_API_KEY, EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN,
                             EXPO_PUBLIC_FIREBASE_PROJECT_ID, EXPO_PUBLIC_FIREBASE_STORAGE_BUCKET,
                             EXPO_PUBLIC_FIREBASE_MESSAGING_SENDER_ID, EXPO_PUBLIC_FIREBASE_APP_ID
           Flutter        → FIREBASE_API_KEY, FIREBASE_AUTH_DOMAIN, FIREBASE_PROJECT_ID,
                             FIREBASE_STORAGE_BUCKET, FIREBASE_MESSAGING_SENDER_ID, FIREBASE_APP_ID
       — Any other client-side SDK init value

  2. .env.local  ← gitignored (add to .gitignore if not already present)
     Contains ONLY truly secret server-side keys with placeholder values.
     Top of file must have this comment:
       # DO NOT commit this file. Add these values to your deployment platform
       # (Vercel → Project Settings → Environment Variables, or Netlify → Site Settings).
     Secret keys go here: STRIPE_SECRET_KEY, OPENAI_API_KEY, database passwords,
     webhook secrets, any key that starts without NEXT_PUBLIC_.

  3. .env.example  ← committed to GitHub
     A copy of ALL variables from both .env AND .env.local, with placeholder values.
     This is the reference file users copy when setting up locally or deploying.
     Top of file: # Copy this file to .env and .env.local and fill in your values.
     Group public vars first, then secret vars, each section with a comment header.

  When writing .gitignore: ensure .env.local and .env*.local are listed.
  Never write real credentials in any file — always placeholder values only.

  Example structure for a Next.js + Firebase + Stripe app:

  .env:
    # Firebase (safe to commit — access controlled by Firebase Security Rules)
    NEXT_PUBLIC_FIREBASE_API_KEY=your-firebase-api-key
    NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project-id.firebaseapp.com
    NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
    NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
    NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your-sender-id
    NEXT_PUBLIC_FIREBASE_APP_ID=your-app-id
    # Stripe publishable key (safe to commit — public by design)
    NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-publishable-key

  .env.local:
    # DO NOT commit this file. Add these to Vercel/Netlify environment variables.
    STRIPE_SECRET_KEY=sk_test_your-stripe-secret-key

  .env.example:
    # Copy this file to .env and .env.local and fill in your values.
    # PUBLIC — safe to commit (put in .env)
    NEXT_PUBLIC_FIREBASE_API_KEY=your-firebase-api-key
    NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project-id.firebaseapp.com
    NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
    NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
    NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your-sender-id
    NEXT_PUBLIC_FIREBASE_APP_ID=your-app-id
    NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-publishable-key
    # SECRET — DO NOT commit (put in .env.local or deployment platform)
    STRIPE_SECRET_KEY=sk_test_your-stripe-secret-key
"""


# ---------------------------------------------------------------------------
# Forgefy UI SDK guidance (feature-flagged)
# ---------------------------------------------------------------------------
# forgefy_ui (Flutter) and @forgefy/ui (React web + native) give generated apps
# layout + animation primitives. This guidance is injected ONLY when
# FORGEFY_UI_ENABLED is truthy: until the packages are published to pub.dev / npm,
# every build's pub-get / npm-install would fail on the missing dependency — and
# unlike the DB SDK this applies to EVERY app, so it stays off by default. Flip
# the env var after publishing.


def _ui_enabled() -> bool:
    from app.config import get_settings
    return get_settings().FORGEFY_UI_ENABLED


_FLUTTER_UI_GUIDANCE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORGEFY UI — layout, animation, SLIVER-FIRST screens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use the `forgefy_ui` package for structure and motion instead of hand-writing
Column/Row/SizedBox or AnimationController.
  • pubspec.yaml → add  forgefy_ui: ^0.1.0
  • Import in UI files (Spacer/Wrap overlap Flutter's — hide them):
      import 'package:flutter/material.dart' hide Spacer, Wrap;
      import 'package:forgefy_ui/forgefy_ui.dart';

SLIVER-FIRST SCREENS — build EVERY scrolling page this way, NOT
Scaffold + SingleChildScrollView + Column:
  SliverScreen(
    gutter: 16,
    appBar: SliverHeader(title: Text(...), pinned: true),
    slivers: [
      SliverGap(12),
      SliverStagger(itemCount: items.length, itemBuilder: (c, i) => ItemCard(items[i])),  // lists
      // grid:           SliverGridView(columns: n, itemCount: .., itemBuilder: ..)
      // fixed section:  SliverBox(child: VStack(spacing: 12, children: [...]))
      // empty/error/loading: SliverFill(child: <state widget>)
    ],
  )

LAYOUT inside boxes: VStack/HStack(spacing:), Grid(columns:), Wrap, Spacer, and
Responsive.value(context, mobile: .., tablet: .., desktop: ..) for adaptive
column counts.
ANIMATION: FadeIn / SlideIn / ScaleIn (entrances), Stagger / SliverStagger
(lists), AnimatedVisibility (show/hide) — prefer these over raw AnimationController.

forgefy_ui owns STRUCTURE + MOTION; the design system still owns VALUES — colors
via AppColors, text via AppTextStyles, spacing from the theme. Never hardcode
colors/sizes into forgefy_ui widgets.
"""

_NEXT_UI_GUIDANCE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORGEFY UI — layout & animation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use `@forgefy/ui/web` for layout/animation instead of hand-rolled flex divs.
  • package.json → add  "@forgefy/ui": "^0.1.0"
  • import { VStack, HStack, Grid, Wrap, Spacer, Scroll, List, Fill, Stagger,
      FadeIn, SlideIn, useResponsiveValue } from "@forgefy/ui/web";
  • Stacks with spacing → <VStack gap={16}> / <HStack gap={8}>.
  • Grid → <Grid columns={useResponsiveValue({ mobile: 1, tablet: 2, desktop: 3 })} gap={12}>.
  • Empty / error / loading → <Fill><EmptyState/></Fill>.
  • Animated lists → <Stagger>…</Stagger>; entrances → <FadeIn>/<SlideIn>.
These use hooks → the files that render them need 'use client'. Tailwind/design
tokens still own color & typography; forgefy_ui owns layout structure.
"""

_RN_UI_GUIDANCE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORGEFY UI — layout & animation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use `@forgefy/ui/native` instead of hand-written StyleSheet flex layout.
  • package.json → add  "@forgefy/ui": "^0.1.0"
  • import { VStack, HStack, Grid, Spacer, List, Fill, Stagger, FadeIn, SlideIn,
      useResponsiveValue } from "@forgefy/ui/native";
  • Screens: VStack/HStack(gap), Grid(columns), Spacer.
  • Long lists → <List data={..} gap={12} renderItem={(item) => <Card .../>} />  (FlatList under the hood).
  • Empty / error / loading → <Fill><EmptyState/></Fill>. Animated lists → <Stagger>.
  • Adaptive values → useResponsiveValue({ mobile, tablet, desktop }).
"""

_UI_GUIDANCE_MAP = {
    "flutter": _FLUTTER_UI_GUIDANCE,
    "next": _NEXT_UI_GUIDANCE,
    "react_native": _RN_UI_GUIDANCE,
}

_UPDATE_UI_RULES = {
    "flutter": """
FORGEFY UI (layout/animation, sliver-first):
  • Use forgefy_ui; build scrolling screens as SliverScreen (CustomScrollView) —
    SliverListView/SliverStagger for lists, SliverGridView for grids, SliverFill
    for empty states, SliverBox(child: VStack(...)) for fixed sections, SliverGap
    for spacing. Inside boxes use VStack/HStack(spacing:)/Grid/Wrap/Spacer and
    Responsive.value(...). Entrances via FadeIn/SlideIn/ScaleIn/Stagger.
  • Add forgefy_ui: ^0.1.0. Import material with `hide Spacer, Wrap;`. Colors/text
    still come from AppColors/AppTextStyles, never hardcoded into forgefy_ui.
""",
    "next": """
FORGEFY UI (layout/animation):
  • Use @forgefy/ui/web — VStack/HStack/Grid/Wrap/Spacer/Scroll/List/Fill/Stagger,
    useResponsiveValue — instead of hand-rolled flex divs. Add "@forgefy/ui": "^0.1.0".
    Files that render them need 'use client'.
""",
    "react_native": """
FORGEFY UI (layout/animation):
  • Use @forgefy/ui/native — VStack/HStack/Grid/Spacer/List/Fill/Stagger,
    useResponsiveValue — instead of StyleSheet flex. Long lists → <List/>; empty
    states → <Fill/>. Add "@forgefy/ui": "^0.1.0".
""",
}


def _update_structure_rules(template_key: str) -> str:
    rules = _UPDATE_STRUCTURE_RULES.get(template_key, "")
    if _ui_enabled():
        rules += _UPDATE_UI_RULES.get(template_key, "")
    return rules


# Tool-usage rules every phase needs. Kept in one constant and appended to each
# phase's system prompt: the design, schema, validator, test and security agents
# each have their own prompt, and guidance that lives in only one of them is
# guidance four agents never see. Byte-stable, so it does not disturb caching.
_TOOL_RULES = """

══════════════════════════════════════════
HOW TO USE THE FILE TOOLS
══════════════════════════════════════════
EDITING — use edit_file, not write_file
  • New file, or a deliberate full rewrite → write_file
  • Any change to a file that already exists → edit_file
  • edit_file replaces one exact string and leaves the rest of the file alone.
    Rewriting a whole file to change a few lines burns output tokens and is how
    long files end up truncated or silently missing code.
  • old_string must match the file exactly, including indentation. read_file
    returns numbered lines — copy the text to replace straight out of it.
  • If edit_file reports an ambiguous match, add surrounding lines to old_string.
    Do not fall back to write_file.

DO NOT RE-READ A FILE YOU JUST EDITED
  • edit_file returns a diff of exactly what changed. That diff is your
    confirmation — you do not need to read the file again to verify it.
  • Read a file ONCE, then make ALL of your changes to it. Repeatedly reading the
    same file between edits wastes the entire phase and changes nothing.
  • Only re-read a file if a tool told you your edit failed.

FINDING THINGS — search before you read
  • grep searches file CONTENTS by regex — use it to find where something is
    defined or used.
  • glob finds files by NAME pattern, e.g. 'src/**/*.tsx'.
  • Opening files one at a time to look for something is the slowest and most
    expensive way to search. Search first, then read only what matched.
  • A project map (when one is provided) lists where symbols are DEFINED — it
    cannot show where they are USED. For any change that spans multiple files,
    or whenever the map does not clearly name the target, run grep first:
      - "update every call site of X"  → grep X, edit each match
      - "where do we call this API?"   → grep the endpoint/method name
      - verifying an edit is complete  → grep for leftovers of the old code
    Trusting a stale map over grep is how edits end up half-applied.
"""


def _with_tool_rules(system: str) -> str:
    """Append the shared tool-usage rules to a phase's system prompt."""
    return system + _TOOL_RULES


def _build_system(template_key: str) -> str:
    structure = _STRUCTURE_MAP.get(template_key, _NEXT_STRUCTURE)
    ui = _UI_GUIDANCE_MAP.get(template_key, "") if _ui_enabled() else ""
    return _with_tool_rules(
        _DESIGN_MANDATE + _BUILD_PREAMBLE + structure + ui + _BUILD_SUFFIX
    )


# ---------------------------------------------------------------------------
# Update agent system prompt
# ---------------------------------------------------------------------------
_UPDATE_SYSTEM = """You are the Forgefy Update Agent. You make precise, targeted changes to existing applications.

══════════════════════════════════════════
DESIGN ENFORCEMENT — read BEFORE implementing
══════════════════════════════════════════
The design system files are your source of truth for all visual decisions.
READ THEM FIRST — before writing any file that contains UI code:
  Flutter    → lib/core/theme/app_colors.dart, lib/core/theme/app_theme.dart
  RN         → src/core/theme/tokens.ts
  Next.js    → app/globals.css, lib/design-system/tokens.ts

HARDCODED VALUES ARE FORBIDDEN in any file you write:
  ✗ Color(0xFF...)  →  ✓ AppColors.primary / AppColors.error / etc.
  ✗ fontSize: 16    →  ✓ AppTextStyles.body
  ✗ padding: 16     →  ✓ AppSpacing.md / spacing.md / var(--space-md)
  ✗ '#6366F1'       →  ✓ colors.primary / var(--color-primary)

USE THE COMPONENT KIT — do not hand-roll standard controls:
  Flutter → shadcn_ui Shad* widgets | Next.js → shadcn/ui in components/ui/ |
  RN → react-native-reusables (components/ui/) if NativeWind, else src/core/components/.
  App-specific composites live in lib/core/widgets/ (Flutter) / components/{feature}/
  (Next) — reuse an existing one before creating a duplicate.

Every user action (tap, submit, toggle) must produce visible feedback.
Use built-in component states (loading, pressed) — don't add extra animation libraries.
══════════════════════════════════════════

══════════════════════════════════════════
MANDATORY WORKFLOW — follow every time
══════════════════════════════════════════
1. Check the RECENT GIT HISTORY and EXISTING PROJECT FILES sections in your task.
   - Git history shows what was already committed. Do NOT redo that work.
   - If a previous run hit the iteration limit, read the already-changed files to
     understand their current state, then continue with the REMAINING work only.
2. Call list_files('.') to confirm the current workspace structure.
3. Read the navigator / router file so you know existing screens and routes.
4. For EVERY file you will write_file to — read it first with read_file.
   This is mandatory, not optional. Reading first prevents overwriting existing logic.
5. Before writing any file that imports from another project file:
   READ that source file first to verify the EXACT class/function/widget names.
   Never assume names — they must match character-for-character.
6. Before using any package import, confirm it appears in INSTALLED PACKAGES.
   If it is missing, add it to pubspec.yaml / package.json FIRST.
7. Implement the remaining (not-yet-done) parts of the request with write_file.
8. After all writes, output a summary starting with DONE: listing every file you wrote
   or modified (use `code` formatting for paths). The validator uses this list.

══════════════════════════════════════════
CRITICAL RULES
══════════════════════════════════════════
- ALWAYS read a file with read_file before writing it. No exceptions.
  Writing without reading first will overwrite code that already works.
- You MUST call write_file at least once. Text descriptions alone fail the task.
- Never output "." or a single word as your response — always write code.
- Never say DONE without having written at least one file.
- If a new screen is added, also update the navigator/router to include it.
- Narrate briefly before each tool call: "Reading the router/layout…", "Writing OnboardingScreen…"
- If you see a file in the git history that was already correctly implemented,
  skip it and move on to what is still missing.
- If your change adds a new third-party service, update all three env files:
  • .env          → add NEXT_PUBLIC_* / Supabase / Firebase client config (pushed)
  • .env.local    → add server-side secrets like STRIPE_SECRET_KEY (gitignored)
  • .env.example  → add ALL new variables from both files so the reference stays complete
  Always use placeholder values — never real credentials.

══════════════════════════════════════════
NO DUPLICATE FILES — CHECK BEFORE CREATING
══════════════════════════════════════════
Your task message includes an "EXISTING PROJECT FILES" list.
BEFORE calling write_file for any file:
  • Check if a file with the EXACT SAME filename already exists anywhere in that list.
  • Check if a file with a SIMILAR name doing the SAME THING exists
    (e.g. auth_service.dart / authentication_service.dart,
         home_page.dart / home_screen.dart,
         userApi.ts / user_service.ts / UserRepository.ts).
  • If either is true → read that existing file first, then write_file to THAT SAME PATH.
  • Only create a brand-new path if you confirm nothing similar already exists.
ONE SOURCE OF TRUTH PER CONCEPT. Never create two files that do the same job.

══════════════════════════════════════════
DEPENDENCY RULE — CRITICAL FOR BUILDS
══════════════════════════════════════════
ANY time you use a package that is not already in package.json / pubspec.yaml,
you MUST update the manifest file FIRST before writing any file that imports it.

Next.js: read package.json, add the package to "dependencies", write package.json.
React Native / Expo: read package.json, add to "dependencies", write package.json.
Flutter: read pubspec.yaml, add under "dependencies:", write pubspec.yaml.

Common packages you must add when you use them:
  shadcn_ui          → shadcn_ui: ^0.55.0            (Flutter control kit — pubspec.yaml)
  nativewind         → "nativewind": "^4.2.0"        (RN — required by react-native-reusables)
  @rn-primitives/*   → "@rn-primitives/<name>": "^1.5.0"  (RN — react-native-reusables primitives)
  lucide-react-native → "lucide-react-native": "^1.25.0"  (RN icons for react-native-reusables)
  @forgefy/client    → "@forgefy/client": "^0.1.0"   (Supabase/Neon DB + auth; Next.js & React Native)
  forgefy_client     → forgefy_client: ^0.1.0        (Supabase/Neon DB + auth; Flutter — pubspec.yaml)
  next-themes        → "next-themes": "^0.3.0"
  framer-motion      → "framer-motion": "^11.0.0"
  lucide-react       → "lucide-react": "^0.400.0"
  @radix-ui/*        → "@radix-ui/<name>": "^1.0.0"
  clsx               → "clsx": "^2.0.0"
  date-fns           → "date-fns": "^3.0.0"
  zustand            → "zustand": "^4.4.0"
  react-hook-form    → "react-hook-form": "^7.0.0"
  zod                → "zod": "^3.22.0"
  axios              → "axios": "^1.6.0"
  @tanstack/react-query → "@tanstack/react-query": "^5.0.0"
  leaflet / react-leaflet → "leaflet": "^1.9.0", "react-leaflet": "^4.2.0"
  mapbox-gl          → "mapbox-gl": "^3.0.0"
  chart.js / react-chartjs-2 → both at latest stable
  recharts           → "recharts": "^2.10.0"
  stripe / @stripe/stripe-js → add both
  firebase           → "firebase": "^10.0.0"
  socket.io-client   → "socket.io-client": "^4.7.0"

If unsure of a version, use "^<major>.0.0" with the current major version.
Never import a package without first confirming it is in the manifest.

══════════════════════════════════════════
FLUTTER PATTERNS
══════════════════════════════════════════
Onboarding screen:
  File: lib/features/onboarding/presentation/pages/onboarding_page.dart
  Use PageView with individual step widgets (icon, title, body, skip/next).
  Store completion in SharedPreferences; check in main.dart to decide initial route.
  Register the route in lib/router/app_router.dart (GoRouter).

Animations:
  Entrance: AnimatedOpacity + SlideTransition triggered in initState via AnimationController.
  List stagger: wrap each item in AnimationBuilder with staggered begin values.
  Page transitions: use PageRouteBuilder with custom transitionsBuilder.
  Hero: wrap shared elements in Hero widget with matching tag.

Dark mode:
  Add ThemeMode state to MaterialApp. Toggle via a provider/bloc. Persist in SharedPreferences.

New screen:
  Create in lib/features/{feature}/presentation/pages/.
  Add the route to lib/router/app_router.dart (GoRouter).

══════════════════════════════════════════
NEXT.JS PATTERNS
══════════════════════════════════════════
CLOUDFLARE DEPLOYMENT RULES (mandatory for all Next.js apps):
  Every API route MUST declare the edge runtime at the top of the file:
    export const runtime = 'edge';
  Every API route MUST use the Web Fetch API, NOT Node.js built-ins:
    ✗  import fs from 'fs'         → NOT available on edge
    ✗  import path from 'path'     → NOT available on edge
    ✗  import crypto from 'crypto' → use: await crypto.subtle.digest(...)
    ✓  fetch(), Request, Response, Headers, URL — all available
    ✓  @vercel/edge, next/server (NextResponse, NextRequest)
  Do NOT import 'server-only' packages (bcrypt, argon2, sharp, prisma with native) in
  edge API routes — use edge-compatible alternatives (jose for JWT, Cloudflare KV for storage).

Onboarding screen:
  File: app/(app)/onboarding/page.tsx
  Multi-step with useState for step index. Store completion in localStorage or cookie.
  Redirect to dashboard if already completed (check in useEffect or middleware).

Animations:
  CSS: add keyframes to globals.css, apply with className.
  Framer Motion: wrap elements in <motion.div> with initial/animate/exit props.
  Tailwind: use transition-*, animate-*, or custom @keyframes in config.

Dark mode:
  Add "next-themes": "^0.3.0" to package.json dependencies FIRST.
  Then wrap <body> in <ThemeProvider> in layout.tsx. Toggle with useTheme hook.

New page:
  Create app/(app)/{feature}/page.tsx. Add link to layout sidebar/nav.

══════════════════════════════════════════
REACT NATIVE PATTERNS
══════════════════════════════════════════
Onboarding screen:
  Route file: app/onboarding.tsx (expo-router). Use FlatList or ScrollView with
  pagingEnabled. Persist completion with react-native-mmkv or expo-secure-store.
  Redirect from app/index.tsx (or app/_layout.tsx) to /onboarding when not completed.

Animations:
  Entrance: Animated.timing with useRef(new Animated.Value(0)).
  List stagger: run Animated.stagger on mount.
  Layout: LayoutAnimation.configureNext before state changes.

New screen:
  Create a route file under app/ (expo-router) — e.g. app/{feature}/index.tsx; register a
  tab with <Tabs.Screen name="..."/> in app/(tabs)/_layout.tsx. Put feature UI/logic in
  src/features/{feature}/ and render it from the thin route file.

══════════════════════════════════════════
IMPORT & SYNTAX RULES — READ BEFORE WRITING ANY FILE
══════════════════════════════════════════

RULE 1 — NEVER import a symbol without reading its source file first.
  Wrong:  import 'package:myapp/core/services/auth_service.dart';  (assume class is AuthService)
  Right:  read_file('lib/core/services/auth_service.dart') → see actual class name → import it

RULE 2 — NEVER use a package not in INSTALLED PACKAGES.
  If you need it, add it to the manifest file FIRST, then import it.

RULE 3 — ALWAYS use the correct import style per framework.
  Flutter:    package imports use 'package:<app_package>/<path>.dart'
              relative imports use '../relative/path.dart'
              NEVER mix the two for the same file.
  Next.js:    use '@/' alias only if tsconfig.json has paths configured for it.
              Check tsconfig.json before using '@/' — fallback to relative paths.
  React Native: use relative paths unless the project has babel module resolver.

RULE 4 — DART/FLUTTER COMMON MISTAKES (fix these before they happen):
  ✗  Color.blue              → ✓  Colors.blue
  ✗  Colors.primary          → ✓  Theme.of(context).colorScheme.primary
  ✗  BorderRadius.circular() → ✓  BorderRadius.circular(n)  (it IS correct — just needs a value)
  ✗  EdgeInsets.only(top=8)  → ✓  EdgeInsets.only(top: 8)   (named args use colon not equals)
  ✗  Widget build() {}       → ✓  Widget build(BuildContext context) {}
  ✗  Text('hi', style: TextStyle(fontSize: 16))  → ✓  Text('hi', style: AppTextStyles.body)
  ✗  setState(() { x = y })  → ✓  setState(() { x = y; })   (semicolon inside lambda)
  ✗  import 'dart:ui' show Color  → ✓  remove — Color is in flutter/material.dart
  ✗  const SizedBox.shrink   → ✓  const SizedBox.shrink()
  ✗  GoRouter(routes: [])    → ✓  GoRouter(routes: <RouteBase>[])  (generic required)
  ✗  context.go('/route')    → only valid if go_router is in pubspec.yaml
  ✗  BlocProvider.of<X>(ctx) → ✓  context.read<X>()  (with flutter_bloc ^8)

RULE 5 — TYPESCRIPT/NEXT.JS COMMON MISTAKES:
  ✗  import X from 'react-query'         → ✓  import { useQuery } from '@tanstack/react-query'
  ✗  import { useState } from 'React'    → ✓  import { useState } from 'react'  (lowercase)
  ✗  export default function() {}        → ✓  export default function PageName() {}  (named)
  ✗  'use client' in a server component  → add 'use client' only in files using hooks/events
  ✗  router.push('/path')  (Next.js 13+) → ✓  use next/navigation's useRouter, not next/router
  ✗  <img src="...">  (Next.js)          → ✓  <Image src="..." width={} height={} from 'next/image'
  ✗  import styles from './X.module.css' → only if that file actually exists — check first
  ✗  API route missing runtime declaration → ✓  add export const runtime = 'edge'; at top of every app/api/*.ts file
  ✗  import fs from 'fs' in API route    → NOT available on Cloudflare edge — use fetch() or Web APIs only

RULE 6 — REACT NATIVE COMMON MISTAKES (this template = Expo Router + Zustand + React Query):
  ✗  import { View } from 'react-native-web'  → ✓  from 'react-native'
  ✗  StyleSheet.create({ x: { color: '#fff' } }) → ✓  use tokens: useAppTheme().colors.xxx
  ✗  import { NavigationContainer } / @react-navigation/*  → ✓  expo-router (routes in app/);
       navigate via import { router } from 'expo-router'
  ✗  configureStore / createSlice / useDispatch (Redux)  → ✓  Zustand create() stores + React Query
  ✗  @react-native-async-storage/async-storage  → ✓  react-native-mmkv or expo-secure-store"""


# ---------------------------------------------------------------------------
# Planner — single text-only API call that returns a structured execution plan
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """You are a software architect. Your ONLY job is to analyse a change request and produce a precise execution plan.

You do NOT write code. You do NOT call tools. You do NOT modify files.

Return ONLY valid JSON — absolutely no other text before or after:
{
  "summary": "<one sentence — what will be built or changed>",
  "skip_design_agent": false,
  "skip_schema_agent": false,
  "skip_test_agent": false,
  "skip_security_agent": false,
  "design_impact": {
    "uses_existing_core_components": ["<component name>"],
    "new_components_needed": ["<component name>"],
    "tokens_used": ["<AppColors.primary>", "<AppTextStyles.headline>"],
    "affects_theme": false,
    "signature_element_appears": false
  },
  "files_to_create": [
    {"path": "<relative/path/to/file>", "purpose": "<what this file contains and does>"}
  ],
  "files_to_modify": [
    {"path": "<relative/path/to/file>", "changes": "<exact changes required — be specific>"}
  ],
  "dependencies": [
    {"package": "<package_name: ^version>", "reason": "<why it is needed>"}
  ],
  "steps": [
    "<atomic step 1>",
    "<atomic step 2>"
  ],
  "constraints": [
    "<anything the executor must NOT change or break>"
  ]
}

Rules:
- All file paths must be relative to the workspace root
- skip_design_agent: set true ONLY when ALL of these are true:
    new_components_needed is empty, affects_theme is false, signature_element_appears is false.
    Pure logic, data, or backend changes qualify. Any visual or UI change must be false.
- skip_schema_agent: set true ONLY when the change stores NO new data — no new entity,
    no new field on an existing entity, no new relationship. Pure UI, styling, copy,
    navigation, or read-only display changes qualify. If the request adds anything the
    app must remember between sessions, it must be false.
- skip_test_agent: set true ONLY when the change adds NO testable logic — pure styling,
    copy edits, asset swaps, or config changes. Any new business rule, calculation,
    validation, state transition, or data mapping must be false.
- skip_security_agent: set true ONLY when the change has NO authentication, NO API keys,
    NO user permissions, NO external API calls, NO data storage (DB, files, cookies).
    Pure UI-only or pure read-only data display changes qualify.
- design_impact.uses_existing_core_components: ALWAYS list which core components will be
  reused — never plan to create a new component if a core one already handles it
- If the request touches ANY visual element, confirm the tokens to be used in tokens_used
- Do NOT plan hardcoded values — if unsure of the token name, write "READ from core/theme"
- Never invent packages that do not exist in the target framework ecosystem
- Each step must be atomic — one conceptual change per step
- Production-quality plan only — no placeholders, no TODOs, no vague steps
- JSON only. Zero prose outside the JSON."""


def _build_planner_msg(
    prompt: str,
    blueprint: dict[str, Any],
    workspace: Path,
    template_key: str,
    app_name: str,
) -> str:
    framework = {
        "flutter": "Flutter/Dart (flutter_bloc)",
        "next": "Next.js / TypeScript / Tailwind CSS",
        "react_native": "React Native / TypeScript / Expo",
    }.get(template_key, template_key)
    try:
        # The project map (paths + key symbols) lets the planner target real
        # files by name/symbol instead of guessing from a bare directory listing.
        file_tree = _build_project_map(workspace)
    except Exception:
        file_tree = "(unavailable)"
    bp_excerpt = json.dumps({
        "features": blueprint.get("features", [])[:10],
        "entities": blueprint.get("entities", [])[:8],
        "description": (blueprint.get("app_description") or blueprint.get("description", ""))[:300],
    })
    # Strip the history preamble that update_worker prepends — the planner only
    # needs the current request, not the conversation history.
    planner_prompt = prompt
    if "\nCURRENT REQUEST\n" in prompt:
        planner_prompt = prompt.split("\nCURRENT REQUEST\n", 1)[-1].strip()
    # Cap so the plan JSON output isn't squeezed by a huge input.
    if len(planner_prompt) > 4000:
        planner_prompt = planner_prompt[:4000]
    return (
        f"App: {app_name}\n"
        f"Framework: {framework}\n\n"
        f"Blueprint excerpt:\n{bp_excerpt}\n\n"
        f"Project map (path: key symbols):\n{file_tree[:3000]}\n\n"
        f"Change request:\n{planner_prompt}"
    )


def _call_planner(
    planner_input: str,
    *,
    backend: str = "claude",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    ollama_model: str = "",
    ollama_timeout: int = 120,
) -> dict[str, Any] | None:
    """Single text-only call to get an execution plan. Returns parsed dict or None on failure."""
    try:
        if backend == "claude":
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model, max_tokens=4096,
                system=_PLANNER_SYSTEM,
                # Sonnet 5+ thinks by default; thinking blocks lead `content`, so the
                # content[0].text read below would hit a ThinkingBlock. JSON-only call.
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": planner_input}],
            )
            raw = resp.content[0].text.strip() if resp.content else ""

        elif backend == "gemini":
            import requests as _req
            url = _GEMINI_URL.format(model=model)
            r = _req.post(
                url, params={"key": api_key},
                json={
                    "system_instruction": {"parts": [{"text": _PLANNER_SYSTEM}]},
                    "contents": [{"role": "user", "parts": [{"text": planner_input}]}],
                    "generationConfig": {"maxOutputTokens": 4096},
                },
                timeout=60,
            )
            r.raise_for_status()
            parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            raw = "".join(p.get("text", "") for p in parts).strip()

        elif backend in ("gpt", "openai"):
            from openai import OpenAI
            client_oai = OpenAI(api_key=api_key)
            resp_oai = client_oai.chat.completions.create(
                model=model, max_tokens=4096,
                messages=[
                    {"role": "system", "content": _PLANNER_SYSTEM},
                    {"role": "user", "content": planner_input},
                ],
            )
            raw = (resp_oai.choices[0].message.content or "").strip()

        else:  # Qwen3 — hosted via OpenRouter, or local Ollama
            from app.ai.qwen import using_openrouter

            if using_openrouter():
                from app.ai.openrouter import PLAN, chat_openrouter
                raw = chat_openrouter(_PLANNER_SYSTEM, planner_input, task=PLAN, max_tokens=4096)
            else:
                import requests as _req

                from app.ai.ollama_http import ollama_headers
                r = _req.post(
                    f"{base_url.rstrip('/')}/api/chat",
                    json={
                        "model": ollama_model,
                        "messages": [
                            {"role": "system", "content": _PLANNER_SYSTEM},
                            {"role": "user", "content": planner_input},
                        ],
                        "stream": False,
                    },
                    headers=ollama_headers(),
                    timeout=ollama_timeout,
                )
                r.raise_for_status()
                raw = (r.json().get("message") or {}).get("content", "").strip()

        # Normalise: strip think tags, markdown fences, extract first {...}
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        raw = m.group(0) if m else raw
        return json.loads(raw)

    except Exception as exc:
        logger.warning("Planner call failed (%s) — proceeding without plan: %s", backend, exc)
        return None


_VALIDATOR_SYSTEM = """You are the Forgefy Validation and Fix Agent. You run AFTER the executor to
CHECK AND FIX any remaining issues in the implementation. You have FULL tool access:
read_file, write_file, delete_file, list_files, analyze_code. Use write_file liberally —
your job is not just to report problems but to FIX every single one you find.
NEVER explain your role. NEVER ask questions. NEVER say you cannot make changes.
Your final output MUST always end with a line starting with "VALIDATED:".

══════════════════════════════════════════
DEMO SCREEN CHECK (run FIRST — cheap and critical)
══════════════════════════════════════════
The template shipped a demo/placeholder home screen. It must NOT remain the app's
landing point:
  1. Open the router/navigator (app/_layout, app/(tabs)/_layout, lib/router/…,
     or MainNavigator) and check what the app opens on.
  2. If it still points at a demo/welcome/showcase screen full of sample content,
     FIX IT: delete or replace that screen with the app's real home screen and
     update the router accordingly.
  3. Also check that screens for features listed in the plan are actually routed —
     an implemented-but-unwired screen is invisible to the user in previews.
  4. Confirm the launch page shows THIS app's real content (per the blueprint) —
     not demo/sample content and not an empty shell. Fix it if not.

══════════════════════════════════════════
DESIGN AUDIT (run BEFORE structural checks)
══════════════════════════════════════════

PASS 1 — Token compliance scan
For each file listed in the plan's files_to_create and files_to_modify, read the file
and search for hardcoded values:
  Flutter: any Color(0xFF...) NOT defined in AppColors, any fontSize: <number> not via
           AppTextStyles, any padding/margin with raw numbers not via AppSpacing
  RN:      any backgroundColor/color with a raw hex string '#...', any fontSize not
           via typography.sizes
  Next.js: any color: '#...' or color: 'rgb(...)' inline styles, any px values for
           spacing not via CSS variable

If found: replace with the correct token. Log:
  "Design fix: replaced hardcoded {value} with {token} in {file}"

PASS 2 — Quality floor check
For each screen/page file in the plan:
  • Does it handle loading state? If not — add AppLoading/Skeleton usage.
  • Does it handle error state?   If not — add error display.
  • Does it handle empty state?   If not — add AppEmptyState/EmptyState.
  • Does AppBar/header exist?     If not — add it.

PASS 3 — Component-kit reuse check
Standard controls must come from the platform kit, NOT custom code:
  • Flutter → shadcn_ui Shad* widgets (ShadButton/ShadInput/ShadCard/…). A
    hand-rolled or raw-Material button/input/card, or a custom AppButton/AppTextField
    used for a standard control, is a violation — replace it with the Shad* widget.
  • Next.js → shadcn/ui components in components/ui/ (@/components/ui/*).
  • RN → react-native-reusables (components/ui/) when NativeWind is set up, else the
    pre-built components in src/core/components/.
Shad* widgets and @/components/ui/* are the canonical components — do NOT flag them as
duplicates or "merge" them into a custom widget. Only genuine app-SPECIFIC composites
belong in lib/core/widgets/ / components/{feature}/; if two of THOSE duplicate each
other, merge into the canonical path and delete_file the duplicate.

Output after design audit:
  "DESIGN AUDIT: Fixed N token violations, N quality floor gaps, N duplicate components"

══════════════════════════════════════════
STRUCTURAL CHECKS (run after design audit)
══════════════════════════════════════════

══════════════════════════════════════════
MANDATORY WORKFLOW
══════════════════════════════════════════
1. Call list_files('.') to see the full current workspace.
2. Read the execution plan to know what was supposed to be built.
3. For EVERY file listed in the plan:
   a. Read the file with read_file.
   b. Check: Does it exist? Is the implementation complete (no TODO/placeholder/pass)?
      Are all imports valid and present in pubspec.yaml or package.json?
      Are all referenced classes/functions/widgets actually defined?
4. Check integration points:
   - Flutter: new screens must be registered in lib/router/app_router.dart (GoRouter)
   - Next.js: new pages must have a link in components/layout/; new packages in package.json
   - React Native: new screens are route files in app/ (expo-router); a new tab is
     registered with <Tabs.Screen> in app/(tabs)/_layout.tsx
5. ── STRUCTURE CHECK (see FOLDER STRUCTURE RULES in your task) ──
   a. For Flutter: verify every new feature has ALL required sub-folders:
      data/datasources/, data/models/, data/repositories/,
      domain/entities/, domain/repositories/, domain/usecases/,
      presentation/bloc/, presentation/pages/, presentation/widgets/
      And that lib/core/injection.dart, lib/router/app_router.dart, lib/main.dart exist
      (there is NO lib/app.dart — the root MaterialApp.router is in main.dart).
   b. For React Native (Expo Router): verify each feature's route files exist under app/
      (e.g. app/{feature}/index.tsx, [id].tsx) and its logic under
      src/features/{feature}/ (components/, hooks/, store.ts [Zustand], api.ts [React Query],
      types.ts). Confirm app/_layout.tsx and app/(tabs)/_layout.tsx exist. There is NO
      src/App.tsx, src/navigation/, Redux store, or slices — flag those if introduced.
   c. For Next.js: verify pages are in app/(app)/, API routes in app/api/, shared
      components in components/ui/ or components/layout/.
   d. If ANY file is in the WRONG location — move it with write_file to the correct
      path, then delete the old file with delete_file. Never leave files out of place.
6. Check for duplicate files (two files doing the same job). If found, merge into
   the canonical path and delete_file the redundant one.
7. Fix EVERY problem found. Do not leave broken code or misplaced files.
8. ── STATIC ANALYSIS (MANDATORY — run this BEFORE finishing) ──
   Call analyze_code() — it runs dart analyze (Flutter) or tsc --noEmit (TS projects).
   Read the output carefully:
   a. If it reports errors: read EVERY file mentioned in the error output and fix each one.
      Common fixes:
        Flutter: wrong class name → check the actual import; missing import → add it;
                 type mismatch → fix the type annotation; undefined getter/method →
                 check the correct property name in the relevant file.
        TS/Next: Property does not exist → check the actual type definition and fix;
                 Cannot find module → add the package or fix the import path;
                 Type '...' is not assignable → fix the type or add a cast.
   b. After fixing, call analyze_code() again to confirm zero errors remain.
   c. If analyze_code() returns an environment/tool error (e.g. "command not found",
      "not the tsc command you are looking for", "No such file", "not recognized"):
      DO NOT give up. Instead: manually read every file listed in the plan, scan for
      obvious syntax errors, wrong imports, missing semicolons, type mismatches, and
      fix each one with write_file. Then proceed to step 9 noting the environment issue.
   d. Only proceed to step 9 when analyze_code() returns "no issues found", "no errors
      found", or when step (c) forced a manual review pass.
9. After all checks, fixes, and a clean analyze_code(), output a report starting with VALIDATED:
   Use markdown: list what was checked, what was fixed/moved, and confirm the build should succeed.

══════════════════════════════════════════
CRITICAL RULES
══════════════════════════════════════════
- Be thorough. Check every file in the plan — not just the ones you wrote.
- Fix silently — do not ask questions, do not leave issues unfixed.
- Do NOT rewrite working code. Only fix actual problems and misplacements.
- analyze_code() is NOT optional. You MUST call it and fix everything it reports.
- If everything is correct: output "VALIDATED: All checks passed — no issues found."
- If you fixed things: output "VALIDATED: Fixed N issue(s) — <brief description>."
- Always include the DESIGN AUDIT summary line before the VALIDATED line.
"""


def _validator_user_msg(
    app_name: str,
    template_key: str,
    plan: dict[str, Any] | None,
    prompt: str,
    workspace: Path | None = None,
    exec_summary: str = "",
) -> str:
    framework = {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}.get(
        template_key, template_key
    )
    plan_section = (
        f"Execution plan to validate:\n{json.dumps(plan, indent=2)}"
        if plan
        else "No structured plan — validate based on the change request below."
    )
    # Only use the core request (strip history preamble if present)
    core_prompt = prompt.split("\nCURRENT REQUEST\n", 1)[-1].strip() if "\nCURRENT REQUEST\n" in prompt else prompt

    # Scope the validator to only the files the executor touched
    exec_section = ""
    if exec_summary:
        exec_section = (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "WHAT THE EXECUTOR DID — focus your checks here\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "The executor reported the following. Validate THESE files first — "
            "read each one, check for correctness, run analyze_code() to catch any "
            "type errors or missing imports, then check integration points.\n\n"
            + exec_summary.strip() + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    structure_rules = _update_structure_rules(template_key)
    structure_section = (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "FOLDER STRUCTURE RULES — validate all files are in the right place\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + structure_rules
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    ) if structure_rules else ""

    workspace_section = ""
    if workspace is not None:
        scan = _scan_workspace(workspace)
        workspace_section = (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CURRENT FILE TREE — check structure and duplicates\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "1. Verify every file is in the correct folder per the FOLDER STRUCTURE RULES above.\n"
            "   If a file is in the wrong place, move it: write_file to correct path, delete_file the old one.\n"
            "2. Check for two files doing the same job "
            "(e.g. home_page.dart AND home_screen.dart, userApi.ts AND user_service.ts).\n"
            "   If duplicates exist, merge into the canonical path and delete_file the redundant one.\n\n"
            + scan + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    return (
        f"App: {app_name}\nFramework: {framework}\n\n"
        f"Original change request:\n{core_prompt[:600]}\n\n"
        f"{plan_section}\n"
        f"{exec_section}"
        f"{structure_section}"
        f"{workspace_section}\n"
        "Validate that every item in the plan was correctly and completely implemented, "
        "every file is in the correct folder per the structure rules, "
        "no duplicates exist, and the code will build without errors. "
        "Fix anything wrong, then output your validation report starting with VALIDATED:"
    )


_DESIGN_AGENT_SYSTEM = """You are the Forgefy UI/UX Design Agent. You run AFTER the planner and BEFORE the executor.
Your job is to ensure the design system is ready, complete, and premium-grade for the features
about to be built, so the executor can focus purely on logic without making visual decisions.

══════════════════════════════════════════
PREMIUM DESIGN STANDARD — NON-NEGOTIABLE
══════════════════════════════════════════
Every component and token you write must meet the standard of premium products like
Apple, Linear, Notion, Stripe, and Arc Browser. This means:

DESIGN LANGUAGE
  • Modern, minimalistic but premium — not flat, not skeuomorphic
  • Inspired by Apple Human Interface Guidelines and modern SaaS design
  • Consistent design system: typography scale, color palette, 4px/8px spacing grid,
    border radius tokens, elevation/shadow system, animation durations and easing curves
  • Component variants for every interactive state

VISUAL QUALITY
  • Smooth gradients where they improve hierarchy — never decorative noise
  • Glassmorphism effects ONLY where they improve depth (e.g. floating nav, modals)
  • Subtle layered shadows — not harsh drop shadows
  • Large touch targets (minimum 44px/44dp)
  • Proper visual hierarchy: primary → secondary → tertiary actions are visually distinct
  • Consistent padding/spacing on a 4px grid across all components
  • Modern cards with elegant, low-opacity shadows
  • Premium empty states: illustration concept + heading + subtext + CTA
  • Skeleton loaders — never bare spinners for content loading

UX REQUIREMENTS
  • Reduce clicks: surface the most common action at thumb reach
  • Improve information hierarchy: most important content gets the most visual weight
  • Add contextual actions — surface them inline, not buried in menus
  • Micro-interactions on every primary action (tap feedback, press scale, submit animation)
  • Haptic feedback for mobile (Flutter: HapticFeedback; RN: Haptics from expo-haptics)
  • Smooth page transitions — no instant cuts
  • State preservation between navigations
  • Accessibility: contrast ratio ≥ 4.5:1, min touch target 44px, semantic labels

ANIMATIONS (implement using framework built-ins — no extra heavy libraries)
  • Page transitions: slide + fade, 250ms, ease-out
  • Hero animations where elements travel between screens
  • Staggered list animations on first render (items slide up + fade in, 50ms stagger)
  • Button press: scale(0.97) + opacity(0.85) on press
  • Modal/bottom sheet: slide up from bottom, 300ms, spring curve
  • Loading states: shimmer skeleton animation (pulse opacity 0.4 → 1.0 → 0.4, 1200ms loop)

COMPONENT STANDARDS
  Every reusable component must have ALL of these variants:
  • Buttons: primary, secondary, ghost, destructive + loading + disabled states
  • Inputs: default, focused, error, disabled + label + hint + character count
  • Cards: elevated, outlined, filled + pressable (with ripple/scale) variant
  • Dialogs: title + body + primary action + secondary action + destructive action
  • Bottom sheets: drag handle + title + scrollable body + safe-area aware
  • Navigation bars: active/inactive states + badges + tooltip labels
  • App bars: transparent-on-scroll, blurred background, with/without back button
  • Chips: filter (toggle), status (non-interactive), action (tappable)
  • Badges: numeric (count), dot (unread indicator), status (colored)
  • Empty states: icon/illustration + heading + subtext + optional CTA
  • Error states: inline (under inputs), banner (top of screen), full-screen (no data)

MOBILE UX (Flutter / React Native)
  • Follow platform conventions (iOS: bottom tabs, material gestures; Android: back gesture)
  • Respect safe areas: wrap in SafeArea / use MediaQuery.padding
  • One-handed usage: primary actions in bottom 60% of screen
  • Bottom sheets over intrusive dialogs for confirmations and pickers
  • Swipe-to-dismiss where contextually appropriate

DARK MODE
  • Full dark mode support with DEDICATED dark tokens — never just invert light colors
  • Dark palette: backgrounds #0F0F0F, surfaces #1C1C1E, surface variant #2C2C2E
  • Text on dark: primary #F2F2F7, secondary #EBEBF0, muted #8E8E93
  • Ensure ≥ 4.5:1 contrast on both light and dark
  • Dark mode glassmorphism: background rgba(28,28,30,0.7) + backdrop-blur

PERFORMANCE
  • Avoid unnecessary widget rebuilds (Flutter: const constructors, RepaintBoundary)
  • Animations at 60fps: use native driver where possible (RN), avoid layout animations
  • Lazy load long lists (ListView.builder / FlatList with windowSize)
  • Skeleton loaders appear instantly — no delay before showing loading state

══════════════════════════════════════════
YOUR MISSION
══════════════════════════════════════════
1. Read the existing design system files (tokens, core components, theme).
2. Upgrade them to meet the premium design standard above where they fall short.
3. Write or extend core components the executor will need for the incoming features.
4. Never implement feature screens or business logic — only design system files.

══════════════════════════════════════════
MANDATORY WORKFLOW
══════════════════════════════════════════
STEP 1 — Read design system files first (ALWAYS, no exceptions)
  Flutter    → read lib/core/theme/app_colors.dart, app_text_styles.dart, app_theme.dart
               list lib/core/widgets/ and read each file
  RN         → read src/core/theme/tokens.ts, src/core/theme/useTheme.ts
               list src/core/components/ and read each file
  Next.js    → read app/globals.css, lib/design-system/tokens.ts
               list components/ui/ and read each file

STEP 2 — Upgrade the design system to premium standard.
  For EACH token file you read:
    • Add missing tokens: glassmorphism surfaces, animation durations/easings,
      dedicated dark mode color tokens, elevation layers (0, 1, 2, 3)
    • Add missing component variants (loading, disabled, error) if any core component lacks them
    • Add stagger animation helpers if the framework allows it
    • Add haptic feedback helpers for mobile

STEP 3 — Analyse the plan's new_components_needed and design_impact.
  For each component the plan says the executor will need:
    a. Check if it already exists in the core components directory.
    b. If it exists — read it. If it doesn't meet the premium standard, upgrade it.
    c. If it doesn't exist — create it now to the premium standard.

STEP 4 — Check and upgrade the signature_element.
  The signature_element from the blueprint must be implemented as SignatureWidget.
  Read the existing SignatureWidget. If it's a generic placeholder or doesn't reflect
  the actual signature_element — rewrite it to be domain-specific and premium.

STEP 5 — Token completeness check.
  Scan the plan's tokens_used list. For each token:
    • Flutter: confirm it exists in AppColors / AppTextStyles / AppSpacing / AppRadius.
    • RN: confirm it exists in colors / typography / spacing / radius in tokens.ts.
    • Next.js: confirm it exists as a CSS variable in globals.css and in tokens.ts.
  If any token is MISSING — add it to the appropriate file.

STEP 6 — Output a design brief.
  Write a summary starting with DESIGN READY: that tells the executor:
  • Which core components are available to import (list them with import paths)
  • Which tokens to use for the new features (list token names)
  • What animations/interactions are available (e.g. "use AppAnimations.stagger")
  • The signature_element implementation status
  • Any files you upgraded or created

══════════════════════════════════════════
WHAT YOU MUST NOT DO
══════════════════════════════════════════
- Do NOT write feature screens, pages, or business logic
- Do NOT implement API routes, state management, or data models
- Do NOT touch navigation/routing files
- Do NOT use hardcoded hex values or raw pixel numbers in any file you write
- Do NOT duplicate a component that already exists — upgrade it instead
- Do NOT add heavy animation libraries (no Lottie unless already in pubspec, no framer-motion
  for micro-interactions — use built-in AnimatedOpacity / Animated.timing / CSS transitions)

══════════════════════════════════════════
TOKEN AND CODE RULES
══════════════════════════════════════════
Flutter:
  • All colors → AppColors.xxx (no Color(0xFF...) outside app_colors.dart)
  • All text → AppTextStyles.xxx (no fontSize: 16 inline)
  • All spacing → AppSpacing.xxx (no EdgeInsets.all(16) raw)
  • All radius → AppRadius.xxx (no BorderRadius.circular(12) raw)
  • Animation helpers → define in AppAnimations class in app_theme.dart

React Native:
  • All colors → colors.xxx from tokens.ts (no '#...' strings inline)
  • All text → typography.sizes.xxx / typography.weights.xxx
  • All spacing → spacing.xxx (no raw numbers for padding/margin)
  • All radius → radius.xxx

Next.js:
  • All colors → var(--color-xxx) via Tailwind class or CSS variable (no '#...' inline)
  • All spacing → var(--space-xxx) or Tailwind space-* classes
  • All radius → var(--radius-xxx) or Tailwind rounded-* classes
  • Animations → CSS transitions + keyframes in globals.css or Tailwind animate-* classes

Every component you write must handle ALL states:
  • Default/idle
  • Loading / skeleton (shimmer animation)
  • Disabled (reduced opacity, non-interactive)
  • Error / destructive (error color, appropriate messaging)
  • Pressed / hover (scale or opacity micro-interaction)
"""


def _design_agent_user_msg(
    app_name: str,
    template_key: str,
    plan: dict[str, Any] | None,
    prompt: str,
    workspace: Path | None = None,
) -> str:
    framework = {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}.get(
        template_key, template_key
    )
    plan_section = (
        f"Execution plan (design_impact tells you what the executor will need):\n{json.dumps(plan, indent=2)}"
        if plan
        else "No structured plan — infer design needs from the change request below."
    )
    core_prompt = (
        prompt.split("\nCURRENT REQUEST\n", 1)[-1].strip()
        if "\nCURRENT REQUEST\n" in prompt
        else prompt
    )

    workspace_section = ""
    if workspace is not None:
        scan = _scan_workspace(workspace)
        workspace_section = (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CURRENT FILE TREE\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + scan + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    return (
        f"App: {app_name}\nFramework: {framework}\n\n"
        f"Incoming change request:\n{core_prompt[:600]}\n\n"
        f"{plan_section}\n"
        f"{workspace_section}\n"
        "Read the design system files, prepare any missing core components, "
        "verify token completeness, then output your design brief starting with DESIGN READY:"
    )


_SCHEMA_SYSTEM = """You are the Forgefy Schema Agent. You run AFTER the planner and BEFORE the design and execution agents.

Your job: turn the blueprint's data model into concrete Postgres/Supabase migration SQL,
so the executor writes queries against a schema that actually exists instead of inventing
table and column names as it goes.

══════════════════════════════════════════
WHAT YOU PRODUCE
══════════════════════════════════════════
1. ONE migration file at supabase/migrations/<UTC timestamp>_<short_snake_name>.sql
   (timestamp format YYYYMMDDHHMMSS — use a plausible current UTC time).
2. A SCHEMA READY: brief listing every table and its columns, for the executor to code against.

You do NOT run migrations — there is no execution path from this workspace to a live
database. You write the SQL file into the repo; applying it is the developer's step.
Say so in your brief rather than claiming tables were created.

══════════════════════════════════════════
BEFORE YOU WRITE
══════════════════════════════════════════
• list_files on supabase/migrations/ — if migrations already exist, READ them first.
  You are extending an existing schema, not redefining it. Never rewrite or delete a
  previous migration; add a new one with ALTER TABLE / CREATE TABLE IF NOT EXISTS.
• Read the blueprint's "entities" — that is the extracted data model. Entities carry
  "fields" (name/type/required/notes) and "relationships" (kind/target).
• If the blueprint has no entities and the plan implies no storage, output
  "SCHEMA READY: no persistent data required — no migration written" and STOP.
  Do NOT invent a schema to look productive.

══════════════════════════════════════════
SQL RULES — non-negotiable
══════════════════════════════════════════
• snake_case, plural table names ("Invoice" entity → invoices table).
• Every table gets: id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now().
  Add updated_at timestamptz when the entity is edited after creation.
• Blueprint field types map as: text→text, integer→integer, decimal→numeric(12,2),
  boolean→boolean, timestamp→timestamptz, date→date, uuid→uuid, json→jsonb,
  enum→text with a CHECK constraint listing the allowed values.
• required: true → NOT NULL. Give NOT NULL columns a DEFAULT when a sensible one exists,
  otherwise leave it and note it in the brief.
• belongs_to → a <target_singular>_id uuid REFERENCES <target_plural>(id) column.
  has_many → the foreign key lives on the OTHER table, not this one.
  many_to_many → a join table named <a>_<b> alphabetically, with both FKs and a
  composite primary key.
• ON DELETE: CASCADE when the child is meaningless without the parent, otherwise RESTRICT.
  Never leave the default (NO ACTION) unstated.
• Add an index on every foreign key column.

══════════════════════════════════════════
ROW LEVEL SECURITY — mandatory
══════════════════════════════════════════
The anon key ships in the client app. RLS IS the security boundary, so a table without
policies is a public table.

• ALTER TABLE <t> ENABLE ROW LEVEL SECURITY; on EVERY table you create.
• Write explicit policies. Default shape for user-owned data:
    - the table carries user_id uuid not null references auth.users(id) on delete cascade
    - select/insert/update/delete policies each using (auth.uid() = user_id)
• For data that is genuinely public-readable, write an explicit
  "select using (true)" policy and say in your brief why it is public.
• NEVER write a policy that is "using (true)" for insert/update/delete.
• If the meeting discussed roles or permissions, model them — do not flatten every
  user to the same access level.

══════════════════════════════════════════
OUTPUT
══════════════════════════════════════════
End with a brief starting with SCHEMA READY: that lists, per table, the exact column
names and types, the FK relationships, and the RLS policy shape. The executor reads
ONLY this brief — if a column is not in it, the executor will not know it exists."""


def _schema_agent_user_msg(
    app_name: str,
    template_key: str,
    blueprint: dict[str, Any],
    plan: dict[str, Any] | None,
    prompt: str,
    workspace: Path | None = None,
) -> str:
    framework = {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}.get(
        template_key, template_key
    )
    entities = blueprint.get("entities") or []
    entities_section = (
        f"Extracted data model ({len(entities)} entities):\n{json.dumps(entities, indent=2)}"
        if entities
        else "The blueprint carries no extracted entities — derive storage needs from the "
        "change request and plan, and write no migration if nothing needs persisting."
    )
    plan_section = (
        f"Execution plan:\n{json.dumps(plan, indent=2)}"
        if plan
        else "No structured plan — infer storage needs from the change request below."
    )
    core_prompt = (
        prompt.split("\nCURRENT REQUEST\n", 1)[-1].strip()
        if "\nCURRENT REQUEST\n" in prompt
        else prompt
    )

    workspace_section = ""
    if workspace is not None:
        workspace_section = (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CURRENT FILE TREE\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + _scan_workspace(workspace) + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    return (
        f"App: {app_name}\nFramework: {framework}\n\n"
        f"Incoming change request:\n{core_prompt[:600]}\n\n"
        f"{entities_section}\n\n"
        f"{plan_section}\n"
        f"{workspace_section}\n"
        "Read any existing migrations, then write the migration and output your brief "
        "starting with SCHEMA READY:"
    )


_TEST_SYSTEM = """You are the Forgefy Test Agent. You run AFTER the validator, once the code compiles.

Your job: prove the feature actually works by writing tests and running them. Static
analysis says the code type-checks; it says nothing about whether it behaves correctly.

══════════════════════════════════════════
WHAT YOU DO
══════════════════════════════════════════
1. Read the changed files to understand what behaviour was added.
2. Write focused tests for that behaviour, in the project's existing test location
   and style — list_files the test directory and read an existing test FIRST.
   If there is no test directory, create the framework's conventional one
   (Flutter → test/, Next.js → __tests__/ or *.test.ts beside the source,
   React Native → __tests__/).
3. Call run_tests() and read the result.
4. If a test fails, decide which side is wrong:
   • The test encodes the wrong expectation → fix the test.
   • The implementation is genuinely broken → fix the implementation.
   Say which one you concluded and why. Do NOT delete or skip a failing test to
   get a green run — a deleted test is a lie about coverage.
5. Repeat until run_tests() passes, or you have a specific reason it cannot.

══════════════════════════════════════════
WHAT TO TEST — and what not to
══════════════════════════════════════════
Test the logic this change introduced:
  • Business rules, calculations, validation, state transitions
  • Data mapping — an entity's fields surviving a round trip
  • Error paths that the code explicitly handles
  • Edge cases the blueprint or plan calls out (empty lists, missing optional
    fields, boundary values)

Do NOT test:
  • The framework itself (that setState re-renders, that a Provider provides)
  • Third-party library internals
  • Static styling — colors, padding, font sizes. These change constantly and a
    test that asserts them breaks on every design tweak while catching no bugs.
  • Trivial getters/setters with no logic

A small number of tests that assert real behaviour beats a large number that
assert the framework works.

══════════════════════════════════════════
HARD RULES
══════════════════════════════════════════
• NEVER weaken a test to make it pass — no commenting out assertions, no skip
  markers, no changing an expected value to whatever the code happened to return.
  If the code is wrong, fix the code.
• Do NOT add a testing framework or dependency that isn't already in the project.
  If none is present, write tests in whatever the framework ships with
  (Flutter: flutter_test; Next.js/React Native: whatever the test script invokes),
  and if there is genuinely no runner, say so and stop rather than installing one.
• Do NOT modify application code except to fix a real defect a test exposed.
  You are not here to refactor.
• If run_tests() reports no suite and there is no runner configured, write the
  tests anyway so they are in the repo, and state clearly that they were not run.

══════════════════════════════════════════
OUTPUT
══════════════════════════════════════════
End with a brief starting with TESTS: that states how many tests you wrote, what
behaviour they cover, the final run_tests() result, and — if anything is still
failing or unrun — exactly what and why. Report the real outcome; a passing claim
that isn't backed by a run_tests() result is worse than admitting the gap."""


def _test_agent_user_msg(
    app_name: str,
    template_key: str,
    plan: dict[str, Any] | None,
    prompt: str,
    workspace: Path | None = None,
    exec_summary: str = "",
) -> str:
    framework = {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}.get(
        template_key, template_key
    )
    plan_section = (
        f"Execution plan (files_to_create / files_to_modify tell you what changed):\n"
        f"{json.dumps(plan, indent=2)}"
        if plan
        else "No structured plan — infer what changed from the request and the summary below."
    )
    core_prompt = (
        prompt.split("\nCURRENT REQUEST\n", 1)[-1].strip()
        if "\nCURRENT REQUEST\n" in prompt
        else prompt
    )
    exec_section = (
        f"\nWhat the executor reported doing:\n{exec_summary.strip()[:1500]}\n"
        if exec_summary
        else ""
    )

    workspace_section = ""
    if workspace is not None:
        workspace_section = (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CURRENT FILE TREE\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + _scan_workspace(workspace) + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    return (
        f"App: {app_name}\nFramework: {framework}\n\n"
        f"Change that was implemented:\n{core_prompt[:600]}\n"
        f"{exec_section}\n"
        f"{plan_section}\n"
        f"{workspace_section}\n"
        "Read an existing test to match the project's style, write tests for the new "
        "behaviour, run them, then output your brief starting with TESTS:"
    )


_SECURITY_SYSTEM = """You are the Forgefy Security Agent. You run AFTER the validator to review the implementation for security vulnerabilities.

══════════════════════════════════════════
SECURITY REVIEW SCOPE
══════════════════════════════════════════
Review ONLY the files that were created or modified in this update.
Focus on these vulnerability categories:

1. INJECTION — SQL injection, NoSQL injection, command injection in API routes
2. AUTHENTICATION — Unprotected routes, missing session checks, weak token handling
3. SENSITIVE DATA EXPOSURE — API keys, secrets, passwords hardcoded in client code
4. INSECURE DIRECT OBJECT REFERENCES — Missing ownership checks before data access
5. XSS — Dangerous innerHTML, dangerouslySetInnerHTML without sanitisation
6. CSRF — Missing CSRF protection on state-changing endpoints
7. INPUT VALIDATION — Missing validation on user-supplied data at API/form boundaries
8. INSECURE STORAGE — Sensitive data stored unencrypted in localStorage or plain files

══════════════════════════════════════════
MANDATORY WORKFLOW
══════════════════════════════════════════
1. Call list_files('.') to orient yourself.
2. For each file in the plan's files_to_create and files_to_modify, read it with read_file.
3. For each vulnerability found: fix it in place with write_file. Do not just report — fix.
4. After all fixes, output a report starting with SECURITY:
   - One line per finding: "Fixed: <type> in <file> — <what was wrong and what changed>"
   - End with: "SECURITY: N issue(s) fixed." or "SECURITY: No issues found."

══════════════════════════════════════════
CRITICAL RULES
══════════════════════════════════════════
- Only review files from the current update — do not audit the entire codebase
- Fix silently — do not ask questions, do not leave exploitable issues unfixed
- Do NOT rewrite working code for non-security reasons
- Only flag real, exploitable issues — not theoretical or style concerns
- If everything is secure: output "SECURITY: No issues found."
"""


def _security_user_msg(
    app_name: str,
    template_key: str,
    plan: dict[str, Any] | None,
    prompt: str,
    workspace: Path | None = None,
) -> str:
    framework = {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}.get(
        template_key, template_key
    )
    plan_section = (
        f"Files created/modified in this update:\n{json.dumps(plan, indent=2)}"
        if plan
        else "No structured plan — review files relevant to the change request below."
    )
    core_prompt = (
        prompt.split("\nCURRENT REQUEST\n", 1)[-1].strip()
        if "\nCURRENT REQUEST\n" in prompt
        else prompt
    )

    workspace_section = ""
    if workspace is not None:
        scan = _scan_workspace(workspace)
        workspace_section = (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CURRENT FILE TREE\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + scan + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    return (
        f"App: {app_name}\nFramework: {framework}\n\n"
        f"Change request:\n{core_prompt[:600]}\n\n"
        f"{plan_section}\n"
        f"{workspace_section}\n"
        "Review the files created/modified in this update for security vulnerabilities. "
        "Fix any issues found with write_file, then output your security report starting with SECURITY:"
    )


def _log_plan(plan: dict[str, Any] | None, log_fn: Callable[[str, str], None] | None) -> None:
    if not log_fn or not plan:
        return
    # Structured event → frontend renders this as an interactive checklist
    log_fn("plan", json.dumps(plan))
    # Human-readable summary in the text log as well
    log_fn("info", f"Plan: {plan.get('summary', 'no summary')}")


def _plan_prefix(plan: dict[str, Any]) -> str:
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "IMPLEMENTATION PLAN — follow this exactly\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{json.dumps(plan, indent=2)}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Execute every item in this plan:\n"
        "• Create all files listed in files_to_create\n"
        "• Apply all changes listed in files_to_modify\n"
        "• Add all packages in dependencies to pubspec.yaml / package.json\n"
        "• Follow steps in order\n"
        "• Production-quality code only — no placeholders, no TODOs\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Adapter resolution (Part L)
#
# BUILD_MODEL is resolved per user at runtime (app/core/build_model.py) across
# ("claude", "Qwen3", "gemini", "gpt"). Every provider difference — wire format,
# cache placement, model chain — lives behind its ProviderAdapter; nothing below
# branches on the model again. This is the ONLY place that maps BUILD_MODEL to
# an adapter.
# ---------------------------------------------------------------------------

_BUILD_MODELS = ("claude", "Qwen3", "gemini", "gpt")


def _build_model(override: str | None = None) -> str:
    """The BUILD_MODEL to use — an explicit per-user override (quota downgrades,
    per-user resolution from app/core/build_model.py) wins over the setting."""
    from app.config import get_settings

    raw = (override or getattr(get_settings(), "BUILD_MODEL", "") or "").strip()
    if raw in ("openai", "openrouter"):  # historical aliases for the gpt chain
        return "gpt"
    return raw if raw in _BUILD_MODELS else "claude"


def _agent_adapter(*, timeout: float | None = None, build_model: str | None = None):
    """Build the ProviderAdapter for the configured BUILD_MODEL.

    ``timeout`` (when the backend supports it) applies to the underlying HTTP
    client — the compile-fix passes use a hard 180s so a hung endpoint cannot
    stall the auto-fix loop.
    """
    import anthropic as _anthropic

    from app.build.provider_loop import (
        AnthropicAdapter,
        GeminiAdapter,
        OllamaAdapter,
        OllamaOpenRouterFallback,
        OpenAIAdapter,
        OpenRouterAdapter,
        unified_loop_enabled,
    )
    from app.config import get_settings

    if not unified_loop_enabled():
        # The legacy per-provider loops were deleted in Part L; the flag is kept
        # so a stale env var is reported rather than silently meaning nothing.
        logger.warning(
            "UNIFIED_AGENT_LOOP is off, but the legacy per-provider loops were "
            "removed — running the unified loop anyway."
        )

    s = get_settings()
    build_model = _build_model(build_model)

    if build_model == "gemini":
        return GeminiAdapter(api_key=s.GEMINI_API_KEY, model=s.GEMINI_MODEL)
    if build_model == "gpt":
        return OpenAIAdapter(api_key=s.OPENAI_API_KEY, model=s.OPENAI_MODEL)
    if build_model == "claude":
        client = _anthropic.Anthropic(api_key=s.ANTHROPIC_API_KEY, timeout=timeout or 600.0)
        return AnthropicAdapter(api_key=s.ANTHROPIC_API_KEY, model=s.ANTHROPIC_MODEL, client=client)

    # Qwen3: Ollama (local daemon or Ollama Cloud) with automatic failover to
    # the OpenRouter CODE chain for the rest of the run once Ollama has burned
    # its retry budget.
    from app.ai.ollama_http import ollama_base_url, ollama_build_model

    ollama = OllamaAdapter(
        base_url=ollama_base_url(s), model=ollama_build_model(s),
        timeout=int(timeout) if timeout else s.OLLAMA_TIMEOUT,
    )
    try:
        from app.ai.qwen import fallback_to_openrouter_enabled

        if fallback_to_openrouter_enabled():
            from app.ai.openrouter import OPENROUTER_BASE_URL, code_models

            key = (s.OPENROUTER_API_KEY or "").strip()
            if key:
                return OllamaOpenRouterFallback(
                    ollama,
                    OpenRouterAdapter(api_key=key, base_url=OPENROUTER_BASE_URL, chain=code_models()),
                )
    except Exception as exc:  # noqa: BLE001 — fallback is best-effort wiring
        logger.warning("qwen: OpenRouter fallback unavailable (%s) — Ollama only", exc)
    return ollama


def run_build_agent(
    workspace: Path,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Run the build agent on the configured BUILD_MODEL; return (summary, tokens).

    One shared loop for every provider: the adapter resolves from BUILD_MODEL
    and supplies only wire format and cache placement (Part L).
    """
    from app.build.provider_loop import run_agent_loop

    system = _build_system(template_key)
    user_msg = (
        f"App name: {app_name}\n"
        f"Template: {template_key}\n\n"
        f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
        "Build this application now following the phases in your instructions. "
        "Narrate each step as you go. When finished, write a user-friendly summary starting with DONE: that describes what was built — screens, features, and anything notable."
    )
    written: list[str] = []
    summary, tokens = run_agent_loop(
        _agent_adapter(), system=system, stable=user_msg, workspace=workspace,
        log_fn=log_fn, cancel_fn=cancel_fn, written=written,
    )
    if written:
        update_project_memory_async(workspace, user_msg, summary, written)
    return summary, tokens






def _tools_to_ollama_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


_SCAN_SKIP_DIRS = frozenset({
    "node_modules", ".next", ".expo", "dist", "build", ".dart_tool",
    "__pycache__", ".git", ".gradle", ".idea", ".vscode",
    "android/.gradle", "ios/Pods", ".pub-cache",
})
_SCAN_SKIP_EXTS = frozenset({
    ".lock", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".ttf", ".otf", ".woff", ".woff2", ".eot", ".mp4", ".mp3",
    ".zip", ".tar", ".gz", ".apk", ".aab", ".ipa", ".so", ".dylib",
})


def _get_git_log(workspace: Path, n: int = 10) -> str:
    """Return the last N git commit messages from the workspace repo.

    This gives the agent ground-truth evidence of what was already implemented
    in previous runs so it can continue rather than re-do work.
    """
    import subprocess as _sp
    try:
        r = _sp.run(
            ["git", "log", "--oneline", f"-{n}", "--no-color"],
            cwd=workspace, capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _get_recent_changed_files(workspace: Path, n: int = 3) -> str:
    """Return the files touched in the last N commits (shows what's already done)."""
    import subprocess as _sp
    try:
        r = _sp.run(
            ["git", "diff", "--name-only", f"HEAD~{n}", "HEAD"],
            cwd=workspace, capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _scan_workspace(workspace: Path, max_files: int = 300) -> str:
    """Return a compact file listing of all source files in the workspace.

    Used to inject project structure into the agent's initial message so it
    can check for existing files before creating new ones.
    """
    lines: list[str] = []
    count = 0
    try:
        for p in sorted(workspace.rglob("*")):
            if count >= max_files:
                lines.append(f"… (truncated at {max_files} files)")
                break
            # Skip hidden dirs and known non-source directories
            parts = p.parts
            if any(
                part.startswith(".") or part in _SCAN_SKIP_DIRS
                for part in parts[len(workspace.parts):]
            ):
                continue
            if p.is_file() and p.suffix.lower() not in _SCAN_SKIP_EXTS:
                lines.append(str(p.relative_to(workspace)))
                count += 1
    except Exception:
        pass
    return "\n".join(lines) if lines else "(empty workspace)"


# ── Project map ──────────────────────────────────────────────────────────────
# A flat path list tells the agent WHICH files exist but not what's in them, so
# it reads file-by-file to locate code. The project map adds each source file's
# top-level symbols (classes, widgets, components, functions, hooks) via cheap
# regex — no LLM calls — so the agent can jump straight to the right file.
_MAP_SOURCE_EXTS = frozenset({".dart", ".ts", ".tsx", ".js", ".jsx"})
_MAP_MAX_FILES = 250
_MAP_MAX_CHARS = 8000
_MAP_MAX_SYMBOLS = 8

# Dart: classes/mixins/enums (widgets, blocs, states, models) + top-level funcs.
_DART_TYPE_RE = re.compile(r"^\s*(?:abstract\s+)?(?:class|mixin|enum)\s+(\w+)", re.M)
_DART_FUNC_RE = re.compile(
    r"^\s*(?:Future<[^>]*>|Stream<[^>]*>|List<[^>]*>|void|Widget|String|int|bool|double)\s+(\w+)\s*\(",
    re.M,
)
# TS/JS: exported functions/classes/consts/interfaces/types/enums (React
# components, hooks, services, prop types) — exports are the public surface.
_TS_EXPORT_RE = re.compile(
    r"^\s*export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|interface|type|enum)\s+(\w+)",
    re.M,
)


def _extract_symbols(path: Path, text: str) -> list[str]:
    """Return up to _MAP_MAX_SYMBOLS top-level symbol names declared in a file."""
    ext = path.suffix.lower()
    names: list[str] = []
    if ext == ".dart":
        names += _DART_TYPE_RE.findall(text)
        names += _DART_FUNC_RE.findall(text)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        names += _TS_EXPORT_RE.findall(text)

    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= _MAP_MAX_SYMBOLS:
            break
    return out


def _build_project_map(workspace: Path) -> str:
    """Return `path: symbol, symbol, …` lines for source files, `path` otherwise.

    Lets the agent locate code by symbol without reading every file. Capped by
    file count and total chars so it stays within the context budget; non-source
    files still appear (path only) so the dedup guidance keeps working.
    """
    lines: list[str] = []
    total = 0
    count = 0
    try:
        for p in sorted(workspace.rglob("*")):
            if count >= _MAP_MAX_FILES:
                lines.append(f"… (truncated at {_MAP_MAX_FILES} files)")
                break
            rel_parts = p.parts[len(workspace.parts):]
            if any(part.startswith(".") or part in _SCAN_SKIP_DIRS for part in rel_parts):
                continue
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix in _SCAN_SKIP_EXTS:
                continue

            rel = p.relative_to(workspace).as_posix()  # forward slashes, matches write_file paths
            if suffix in _MAP_SOURCE_EXTS:
                try:
                    symbols = _extract_symbols(p, p.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    symbols = []
                line = f"{rel}: {', '.join(symbols)}" if symbols else rel
            else:
                line = rel

            if total + len(line) + 1 > _MAP_MAX_CHARS:
                lines.append("… (truncated — map size limit reached)")
                break
            lines.append(line)
            total += len(line) + 1
            count += 1
    except Exception:
        pass
    return "\n".join(lines) if lines else "(empty workspace)"


def _get_installed_packages(workspace: Path) -> str:
    """Read pubspec.yaml or package.json and return the installed dependency names.

    Injected into the executor prompt so the agent never imports a package
    that isn't actually installed.
    """
    # Flutter
    pubspec = workspace / "pubspec.yaml"
    if pubspec.exists():
        try:
            text = pubspec.read_text(encoding="utf-8", errors="replace")
            # Grab only the dependencies / dev_dependencies sections
            lines: list[str] = []
            in_deps = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped in ("dependencies:", "dev_dependencies:"):
                    in_deps = True
                    lines.append(stripped)
                    continue
                if in_deps:
                    # Top-level key signals end of section
                    if line and not line[0].isspace():
                        in_deps = False
                        continue
                    if stripped and not stripped.startswith("#"):
                        lines.append("  " + stripped)
            return "pubspec.yaml dependencies:\n" + "\n".join(lines) if lines else ""
        except Exception:
            pass

    # Next.js / React Native
    pkg = workspace / "package.json"
    if pkg.exists():
        try:
            import json as _json
            data = _json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if deps:
                names = "\n".join(f"  {k}: {v}" for k, v in sorted(deps.items()))
                return f"package.json dependencies:\n{names}"
        except Exception:
            pass

    return ""


def _update_user_msg(
    app_name: str,
    template_key: str,
    blueprint: dict[str, Any],
    prompt: str,
    plan: dict[str, Any] | None,
    workspace: Path | None = None,
) -> str:
    framework = {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}.get(template_key, template_key)
    prefix = _plan_prefix(plan) if plan else ""

    # Inject the canonical folder structure so the agent puts files in the right place
    structure_rules = _update_structure_rules(template_key)
    structure_section = (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "MANDATORY FOLDER STRUCTURE — all new files must follow this\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + structure_rules
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    ) if structure_rules else ""

    file_listing = ""
    if workspace is not None:
        scan = _build_project_map(workspace)
        git_log = _get_git_log(workspace)
        recent_files = _get_recent_changed_files(workspace)
        installed_pkgs = _get_installed_packages(workspace)

        git_section = ""
        if git_log:
            git_section = (
                "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "RECENT GIT HISTORY — what has already been committed\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "These commits are ALREADY in the repo. Do NOT re-implement work that is done.\n"
                "If the previous run hit an iteration limit, READ these files to understand\n"
                "the current state, then continue from where the work stopped.\n\n"
                + git_log + "\n"
            )
            if recent_files:
                git_section += f"\nFiles changed in recent commits:\n{recent_files}\n"
            git_section += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        pkg_section = (
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "INSTALLED PACKAGES — ONLY import from these\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "These are the packages already in the manifest. "
            "You MUST NOT import any package not in this list without first adding it to "
            "pubspec.yaml / package.json.\n\n"
            + installed_pkgs + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        ) if installed_pkgs else ""

        file_listing = (
            git_section
            + pkg_section
            + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "PROJECT MAP — every file and its key symbols\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Each line is `path: symbols` — the classes, widgets, components, functions and\n"
            "hooks defined in that file. USE THIS MAP to find the right file to change. Do\n"
            "NOT read files one by one to discover where things are — the map already tells\n"
            "you. Only call read_file when you need a specific file's full contents to edit\n"
            "it, and prefer reading the 1–3 files the map shows are relevant.\n"
            "The map shows where symbols are DEFINED, not every place they are USED — for\n"
            "multi-file changes, or anything the map doesn't clearly name, use grep first\n"
            "(see HOW TO USE THE FILE TOOLS).\n\n"
            "Before calling write_file for a NEW file:\n"
            "  1. Scan the map for a file with the SAME NAME or SAME PURPOSE\n"
            "     (e.g. auth_service.dart ≈ authentication_service.dart,\n"
            "      home_screen.dart ≈ home_page.dart, userApi.ts ≈ user_service.ts).\n"
            "  2. If found → edit THAT existing path. NEVER create a second file that does\n"
            "     the same job as an existing one.\n\n"
            + scan + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    return (
        f"{prefix}"
        f"App name: {app_name}\n"
        f"Framework: {framework}\n"
        f"Existing blueprint context:\n{json.dumps(blueprint, indent=2)}\n"
        f"{structure_section}"
        f"{file_listing}\n"
        f"User's update request: {prompt}\n\n"
        "Apply this change now. Use list_files('.') to explore the workspace, read the relevant files, "
        "implement every part of the request with write_file — placing every file in the correct "
        "folder per the MANDATORY FOLDER STRUCTURE above, "
        "then write a markdown-formatted summary starting with DONE: that describes exactly what was changed. "
        "Use **bold** for feature names, bullet lists for multiple changes, and `code` for file paths and package names."
    )


def _update_user_msg_with_brief(
    app_name: str,
    template_key: str,
    blueprint: dict[str, Any],
    prompt: str,
    plan: dict[str, Any] | None,
    workspace: Path | None,
    design_brief: str,
    schema_brief: str = "",
) -> str:
    """Like _update_user_msg but prepends the design and schema agents' output so the
    executor knows exactly which components, tokens, tables and columns were prepared."""
    base = _update_user_msg(app_name, template_key, blueprint, prompt, plan, workspace)
    blocks = ""
    if schema_brief:
        blocks += (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "DATABASE SCHEMA BRIEF — query these tables and columns, invent no others\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + schema_brief.strip() + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )
    if design_brief:
        blocks += (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "DESIGN SYSTEM BRIEF — use these components and tokens\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + design_brief.strip() + "\n"
            + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        )
    return blocks + base


# Per-agent iteration budgets — focused agents get fewer iterations to save tokens
_ITERS_DESIGN = 30
_ITERS_SCHEMA = 20
_ITERS_EXEC = 50
_ITERS_VALIDATE = 25
_ITERS_TEST = 25
_ITERS_SECURITY = 20
_ITERS_FIX = 100


def _should_skip_design(plan: dict[str, Any] | None) -> bool:
    if not plan:
        return False
    if plan.get("skip_design_agent"):
        return True
    di = plan.get("design_impact") or {}
    return (
        not di.get("new_components_needed")
        and not di.get("affects_theme")
        and not di.get("signature_element_appears")
    )


def _should_skip_schema(plan: dict[str, Any] | None, blueprint: dict[str, Any]) -> bool:
    """Skip the schema phase only when the planner says nothing is stored.

    Unlike _should_skip_design there is no heuristic fallback: a missing plan means
    run the phase. Guessing wrong the other way leaves the executor writing queries
    against tables that were never defined, which fails at runtime rather than at
    build time.
    """
    if plan and plan.get("skip_schema_agent"):
        # A planner "skip" is overridden when the blueprint carries an explicit data
        # model — entities in the blueprint are direct evidence that something is stored.
        return not (blueprint.get("entities") or [])
    return False


def _should_skip_test(plan: dict[str, Any] | None) -> bool:
    """Skip the test phase only when the planner says nothing testable changed.

    No plan means run it. There is no blueprint-level evidence to override with
    (unlike the schema gate) — the plan is the only signal about whether the
    change introduced behaviour worth asserting on.
    """
    if not plan:
        return False
    return bool(plan.get("skip_test_agent"))


def _should_skip_security(plan: dict[str, Any] | None) -> bool:
    if not plan:
        return False
    return bool(plan.get("skip_security_agent"))


# Phases that review rather than build, and so must file a structured report.
_REPORTING_PHASES = ("validate", "test", "security")
# The tool set those phases get: everything, plus the reporting tool.
_REVIEW_TOOLS = [*TOOLS, REPORT_FINDINGS_TOOL]

# Appended to a reviewing phase's system prompt. The instruction has to be
# explicit and last, because the whole fix-pass decision now rests on the call
# being made — a phase that forgets is treated as unresolved.
_REPORT_INSTRUCTION = """

══════════════════════════════════════════
YOU MUST REPORT YOUR RESULT
══════════════════════════════════════════
Before your final message, call report_findings exactly once.
  • Nothing needs fixing → status='clean', findings=[]
  • Something is genuinely broken → status='issues_found' with one entry per
    problem, each naming the file it is in.
  • Anything you already fixed during this phase is NOT a finding. Report the
    state of the code as you are leaving it, not the problems you solved.
This call, not your prose, decides whether an expensive fix pass runs. Failing to
call it is treated as an unresolved failure.
"""


def _needs_fix_pass(report: dict[str, Any]) -> bool:
    """Return True when the validator's structured report says work remains.

    Reads a field the validator set deliberately, rather than sniffing its prose
    for a "VALIDATED:" prefix. The old test called a clean review dirty whenever
    the model phrased its sign-off naturally, signed off with DONE, or returned
    an empty summary — each of which spent a 100-iteration fix pass on code that
    was already correct.
    """
    return report.get("status") != "clean"


def _format_findings(findings: list[dict[str, Any]]) -> str:
    """Render findings as an ordered, addressable list for the fix pass."""
    if not findings:
        return "(no individual findings were listed)"
    order = {sev: i for i, sev in enumerate(_FINDING_SEVERITIES)}
    ranked = sorted(findings, key=lambda f: order.get(f.get("severity", ""), 99))
    lines = []
    for i, finding in enumerate(ranked, 1):
        where = finding.get("file") or "(file not specified)"
        if line := finding.get("line"):
            where += f":{line}"
        lines.append(
            f"{i}. [{finding.get('severity', 'medium')}] {where} — {finding.get('summary', '')}"
        )
    return "\n".join(lines)


def _make_fix_prompt(report: dict[str, Any]) -> str:
    findings = report.get("findings") or []
    summary = report.get("summary") or ""
    return (
        "The validator reviewed the implementation and found unresolved issues.\n\n"
        f"Validator summary: {summary}\n\n"
        f"Findings to fix:\n{_format_findings(findings)}\n\n"
        "Fix every finding above. Each one names the file it is in — read that file "
        "first, then apply the minimum change needed. Work through them in order and "
        "do not re-explain the issues, just fix them."
    )


# A phase runner: (system_prompt, user_msg, max_iterations) -> (summary, tokens).
# Every backend supplies one by closing over its own tool loop; the phase sequence
# below is then shared, so a new phase is added in exactly one place.
RunPhase = Callable[[str, str, int], tuple[str, int]]


def _run_update_sequence(
    run_phase: RunPhase,
    *,
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    plan: dict[str, Any] | None,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """schema? → design? → execute → validate → fix? → security?

    The single definition of the update pipeline. Backends differ only in how a
    phase is executed (which tool loop, which client args), which is what
    ``run_phase`` abstracts — planning happens in the caller because its
    arguments are backend-specific.
    """
    total_tokens = 0

    # Every phase gets the shared tool-usage rules. Wrapping here rather than at
    # each call site means a new phase cannot be added without them.
    def _phase(system: str, user_msg: str, iters: int) -> tuple[str, int]:
        return run_phase(_with_tool_rules(system), user_msg, iters)

    def _review_phase(
        name: str, system: str, user_msg: str, iters: int,
    ) -> tuple[str, int, dict[str, Any]]:
        """Run a reviewing phase and collect its structured report.

        The report travels through a per-workspace store rather than the return
        value because ``run_phase`` returns (summary, tokens) across five
        backends; changing that contract would mean changing all five.
        """
        reset_report(workspace)
        summary, tokens = run_phase(
            _with_tool_rules(system) + _REPORT_INSTRUCTION, user_msg, iters, _REVIEW_TOOLS,
        )
        report = take_report(workspace) or missing_report(name)
        findings = report.get("findings") or []
        logger.info(
            "phase=%s status=%s findings=%d reported=%s",
            name, report.get("status"), len(findings), report.get("reported"),
        )
        if log_fn and not report.get("reported"):
            log_fn("warning", f"{name.title()} phase ended without a result — treating as unresolved.")
        return summary, tokens, report

    def _cancelled() -> bool:
        return bool(cancel_fn and cancel_fn())

    if _cancelled():
        return "Stopped by user.", total_tokens

    # Pre-execution phases produce briefs the executor reads. Their user messages
    # are built lazily — each one scans the workspace, which is wasted work for a
    # phase the planner told us to skip.
    briefs = {"schema": "", "design": ""}
    pre_exec = (
        (
            "schema", "Database schema",
            _should_skip_schema(plan, blueprint), _SCHEMA_SYSTEM, _ITERS_SCHEMA,
            lambda: _schema_agent_user_msg(app_name, template_key, blueprint, plan, prompt, workspace),
            "Preparing database schema…",
            "Schema phase skipped — no new data is stored.",
        ),
        (
            "design", "Design system",
            _should_skip_design(plan), _DESIGN_AGENT_SYSTEM, _ITERS_DESIGN,
            lambda: _design_agent_user_msg(app_name, template_key, plan, prompt, workspace),
            "Preparing design system…",
            "Design phase skipped — no new components or theme changes.",
        ),
    )

    for key, label, skip, system, iters, make_msg, thinking, skip_msg in pre_exec:
        if skip:
            if log_fn:
                log_fn("info", skip_msg)
            continue
        if log_fn:
            log_fn("info", f"━━━ {label} phase ━━━")
            log_fn("thinking", thinking)
        briefs[key], tokens = _phase(system, make_msg(), iters)
        total_tokens += tokens
        if _cancelled():
            return briefs[key] or "Stopped by user.", total_tokens

    if log_fn:
        log_fn("info", "━━━ Execution phase ━━━")
    exec_summary, tokens = _phase(
        _UPDATE_SYSTEM,
        _update_user_msg_with_brief(
            app_name, template_key, blueprint, prompt, plan, workspace,
            briefs["design"], briefs["schema"],
        ),
        _ITERS_EXEC,
    )
    total_tokens += tokens
    if _cancelled():
        return exec_summary or "Stopped by user.", total_tokens

    if log_fn:
        log_fn("info", "━━━ Validation phase ━━━")
        log_fn("thinking", "Validating implementation…")
    val_summary, tokens, val_report = _review_phase(
        "validate",
        _VALIDATOR_SYSTEM,
        _validator_user_msg(app_name, template_key, plan, prompt, workspace, exec_summary),
        _ITERS_VALIDATE,
    )
    total_tokens += tokens
    if _cancelled():
        return val_summary or exec_summary, total_tokens

    if _needs_fix_pass(val_report):
        found = len(val_report.get("findings") or [])
        if log_fn:
            log_fn("info", "━━━ Post-validation fix pass ━━━")
            log_fn(
                "thinking",
                f"Validator found {found} unresolved issue(s) — running targeted fix pass…"
                if found else "Validator did not sign off — running targeted fix pass…",
            )
        # The fix pass replaces the validator's summary — it is the newer account
        # of the same work.
        val_summary, tokens = _phase(_UPDATE_SYSTEM, _make_fix_prompt(val_report), _ITERS_FIX)
        total_tokens += tokens
        if _cancelled():
            return val_summary or exec_summary, total_tokens
    elif log_fn:
        log_fn("info", "Validation clean — no fix pass needed.")

    # Tests run after the fix pass, so they assert against code that compiles —
    # a test suite written against known-broken code just re-reports the same
    # failure the validator already found.
    if not _should_skip_test(plan):
        if log_fn:
            log_fn("info", "━━━ Test phase ━━━")
            log_fn("thinking", "Writing and running tests…")
        # The test brief is surfaced to the user rather than fed forward: it is
        # evidence about the work, not input to a later phase.
        test_summary, tokens, _test_report = _review_phase(
            "test",
            _TEST_SYSTEM,
            _test_agent_user_msg(app_name, template_key, plan, prompt, workspace, exec_summary),
            _ITERS_TEST,
        )
        total_tokens += tokens
        if test_summary.strip() and log_fn:
            log_fn("info", test_summary.strip()[:500])
        if _cancelled():
            return val_summary or exec_summary, total_tokens
    elif log_fn:
        log_fn("info", "Test phase skipped — no testable logic changed.")

    if not _should_skip_security(plan):
        if log_fn:
            log_fn("info", "━━━ Security review phase ━━━")
            log_fn("thinking", "Reviewing for security issues…")
        # The security agent fixes what it finds in place; its prose isn't
        # surfaced, so only its token cost and finding count are kept.
        _, tokens, sec_report = _review_phase(
            "security",
            _SECURITY_SYSTEM,
            _security_user_msg(app_name, template_key, plan, prompt, workspace),
            _ITERS_SECURITY,
        )
        total_tokens += tokens
        if log_fn and (sec_findings := len(sec_report.get("findings") or [])):
            log_fn("info", f"Security review: {sec_findings} issue(s) addressed.")
    elif log_fn:
        log_fn("info", "Security phase skipped — no auth/API/storage changes.")

    return val_summary or exec_summary, total_tokens


def run_update_agent(
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
    build_model: str | None = None,
) -> tuple[str, int]:
    """Plan → design? → execute → validate → security? on the configured BUILD_MODEL.

    ``build_model`` overrides the setting per user (quota downgrades)."""
    from app.build.provider_loop import run_agent_loop
    from app.config import get_settings

    effective_model = _build_model(build_model)
    s = get_settings()

    if log_fn:
        log_fn("thinking", "Planning changes…")
    # Planner creds per backend: only the text-only planner call branches here;
    # every tool-using phase below runs on the adapter.
    if effective_model == "gemini":
        plan = _call_planner(
            _build_planner_msg(prompt, blueprint, workspace, template_key, app_name),
            backend="gemini", api_key=s.GEMINI_API_KEY, model=s.GEMINI_MODEL,
        )
    elif effective_model == "gpt":
        plan = _call_planner(
            _build_planner_msg(prompt, blueprint, workspace, template_key, app_name),
            backend="gpt", api_key=s.OPENAI_API_KEY, model=s.OPENAI_MODEL,
        )
    elif effective_model == "claude":
        plan = _call_planner(
            _build_planner_msg(prompt, blueprint, workspace, template_key, app_name),
            backend="claude", api_key=s.ANTHROPIC_API_KEY, model=s.ANTHROPIC_MODEL,
        )
    else:  # Qwen3 — routes to OpenRouter's PLAN chain or Ollama internally
        plan = _call_planner(
            _build_planner_msg(prompt, blueprint, workspace, template_key, app_name),
            backend="Qwen3",
        )
    _log_plan(plan, log_fn)
    # The planner call may run on a text-only model; its (small) token usage
    # isn't counted — the tool-loop phases below are the bulk of the cost.

    adapter = _agent_adapter(build_model=build_model)

    def run_phase(
        system: str, user_msg: str, iters: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, int]:
        return run_agent_loop(
            adapter, system=system, stable=user_msg, workspace=workspace,
            log_fn=log_fn, cancel_fn=cancel_fn, max_iterations=iters,
            tools=tools,
        )

    return _run_update_sequence(
        run_phase, workspace=workspace, prompt=prompt, blueprint=blueprint,
        app_name=app_name, template_key=template_key, plan=plan,
        log_fn=log_fn, cancel_fn=cancel_fn,
    )


# ---------------------------------------------------------------------------
# Gemini agent loop
# ---------------------------------------------------------------------------

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Gemini 2.5 Flash has a 1M-token context, so the old 1,500-char cap was starving
# it for no reason: a stylesheet or a component came back as a fragment, and the
# agent spent its whole phase paging through the same file with read_file offsets
# trying to see the rest. Use the same large-context cap as the Claude path.
# Rolling window, in model/tool-response pairs, so a long phase stops resending
# every earlier turn.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_GEMINI_MAX_RETRIES = 4
_GEMINI_BASE_DELAY_S = 5.0

# Client-side pacing, learned from 429s rather than configured. An agent phase
# fires requests back to back with no gap, so even a single user can trip a
# per-minute limit within seconds — it is our own request rate, not anyone
# else's load. Starts at zero so a key with headroom pays nothing for this; the
# first rate-limit response sets it, and it decays once requests get through
# cleanly. Process-local and advisory: worker concurrency is not accounted for.
_gemini_min_interval_s = 0.0
_gemini_last_request_at = 0.0
_GEMINI_MAX_INTERVAL_S = 30.0


def _gemini_pace() -> None:
    """Sleep just long enough to honour the learned minimum request interval."""
    global _gemini_last_request_at
    if _gemini_min_interval_s > 0:
        wait = _gemini_min_interval_s - (time.monotonic() - _gemini_last_request_at)
        if wait > 0:
            logger.debug("Gemini pacing: sleeping %.1fs before next request", wait)
            time.sleep(wait)
    _gemini_last_request_at = time.monotonic()


def _gemini_note_rate_limit(retry_delay: float | None) -> None:
    """Raise the pacing interval after a rate-limit response."""
    global _gemini_min_interval_s
    # Google's own RetryInfo is authoritative when present. The fallback is a
    # guess at a typical free-tier allowance (~10 requests/minute) used only
    # until a response tells us better.
    target = retry_delay if retry_delay and retry_delay > 0 else 6.0
    _gemini_min_interval_s = min(max(_gemini_min_interval_s, target), _GEMINI_MAX_INTERVAL_S)


def _gemini_note_success() -> None:
    """Relax pacing gradually while requests are getting through."""
    global _gemini_min_interval_s
    if _gemini_min_interval_s > 0:
        _gemini_min_interval_s = max(0.0, _gemini_min_interval_s * 0.9 - 0.1)


def _gemini_error_details(resp: Any) -> tuple[float | None, list[str], str]:
    """Pull (retry_delay_seconds, quota_ids, message) out of a Gemini error body.

    Defensive: an error body that does not match the documented shape yields
    empty values rather than raising inside the retry loop.
    """
    try:
        error = (resp.json() or {}).get("error") or {}
    except ValueError:
        return None, [], resp.text[:300]

    retry_delay: float | None = None
    quota_ids: list[str] = []
    for detail in error.get("details") or []:
        kind = str(detail.get("@type", ""))
        if kind.endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                retry_delay = float(raw)
            except ValueError:
                retry_delay = None
        elif kind.endswith("QuotaFailure"):
            quota_ids += [
                str(v.get("quotaId") or v.get("quotaMetric") or "")
                for v in detail.get("violations") or []
            ]
    return retry_delay, quota_ids, str(error.get("message") or "")[:300]


def _gemini_post_with_retry(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    log_fn: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """POST to Gemini, retrying with backoff on rate-limit/overload responses.

    Unlike the Anthropic/OpenAI SDKs (which retry 429/5xx automatically), this
    loop calls the Gemini REST API directly via `requests`, so a single
    transient rate-limit or overload response used to kill the entire build
    task immediately — no second chance. This gives it the same resilience.
    """
    import requests as _req

    delay = _GEMINI_BASE_DELAY_S
    last_exc: Exception | None = None
    told_user = False
    hit_rate_limit = False

    for attempt in range(_GEMINI_MAX_RETRIES + 1):
        _gemini_pace()
        try:
            resp = _req.post(url, params={"key": api_key}, json=payload, timeout=timeout)
        except _req.exceptions.RequestException as exc:
            last_exc = exc
            # A connection error is a local/network problem, never someone
            # else's load — say so rather than blaming demand.
            if log_fn and not told_user:
                told_user = True
                # Provider-agnostic: users never see which backend is serving.
                log_fn("warning", "The AI service is briefly unreachable — checking the connection…")
            logger.warning("Gemini unreachable for %s", url)
        else:
            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                resp.raise_for_status()
                # Only a clean first-try success is evidence the limit has
                # lifted. A success that happened *because* we just backed off
                # is evidence the backoff worked, so it must not undo it.
                if not hit_rate_limit:
                    _gemini_note_success()
                return resp.json()

            retry_delay, quota_ids, message = _gemini_error_details(resp)
            quota_blob = " ".join(quota_ids).lower()

            if resp.status_code == 429:
                # Depleted prepaid credits: retrying returns the identical
                # response every time, so fail immediately with the real fix.
                if "credit" in resp.text.lower():
                    raise RuntimeError(
                        "Gemini prepayment credits are depleted. Add credits at "
                        "https://ai.studio/projects, or switch BUILD_MODEL to "
                        "'claude' or 'gpt' in .env."
                    )
                # A per-DAY quota will not clear within this build either.
                if "perday" in quota_blob.replace("_", ""):
                    raise RuntimeError(
                        "Gemini daily request quota is exhausted for this API key "
                        f"({', '.join(quota_ids) or 'per-day limit'}). It resets at "
                        "midnight Pacific. Enable billing on the key, or switch "
                        "BUILD_MODEL to 'claude' or 'gpt' in .env."
                    )
                # Per-minute limit: genuinely transient, and it is our own
                # request rate. Pace subsequent calls so we stop causing it.
                hit_rate_limit = True
                _gemini_note_rate_limit(retry_delay)
                if log_fn and not told_user:
                    told_user = True
                    # Provider-agnostic for users; the billing hint is operator
                    # information and stays in the log below.
                    log_fn(
                        "warning",
                        "The AI is in high demand right now — pacing requests to stay reliable.",
                    )
                logger.info(
                    "Gemini 429 (%s) — pacing to %.1fs between requests: %s "
                    "(hint: enabling billing on the API key removes this limit)",
                    ", ".join(quota_ids) or "rate limit", _gemini_min_interval_s, message,
                )
            elif log_fn and not told_user:
                # 5xx really is Google's problem — the original wording fits here.
                told_user = True
                log_fn("thinking", "High demand right now — taking a short pause, hang tight…")

            last_exc = _req.exceptions.HTTPError(
                f"{resp.status_code} {resp.reason} for {url}: {message or resp.text[:300]}"
            )
            if retry_delay and retry_delay > 0:
                # Google told us exactly how long to wait; guessing is worse.
                delay = max(delay, retry_delay)

        if attempt < _GEMINI_MAX_RETRIES:
            logger.info(
                "Gemini retry %d/%d in %.0fs", attempt + 1, _GEMINI_MAX_RETRIES, delay
            )
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Gemini agent request failed after {_GEMINI_MAX_RETRIES} retries: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Fix agent — executor-only, no pipeline.
# Used by build_worker's auto-fix loop to avoid running 5 agents per compile error.
# ---------------------------------------------------------------------------

def run_fix_agent(
    workspace: Path,
    prompt: str,
    app_name: str,
    template_key: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Executor-only fix pass on the configured BUILD_MODEL — no planner, design,
    validator, or security."""
    from app.build.provider_loop import run_agent_loop

    return run_agent_loop(
        _agent_adapter(timeout=180.0), system=_UPDATE_SYSTEM, stable=prompt,
        workspace=workspace, log_fn=log_fn, cancel_fn=cancel_fn,
        max_iterations=_ITERS_FIX,
    )


# ---------------------------------------------------------------------------
# Core loop (shared Anthropic streaming turn — used by AnthropicAdapter)
# ---------------------------------------------------------------------------

def _stream_turn(
    client: anthropic.Anthropic,
    model: str,
    system_blocks: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    log_fn: Callable[[str, str], None] | None,
    tool_blocks: list[dict[str, Any]],
) -> Any:
    """Run one streamed Anthropic turn and return the accumulated final message.

    Streaming is what keeps the build feed alive: on a non-streamed call the UI
    sits silent for the whole generation and then receives the turn in one lump.
    Text is flushed at sentence / newline / 120-char boundaries and thinking at
    120 chars, matching the Ollama path so the feed reads identically across
    providers.
    """
    global _thinking_supported

    def _run(thinking: dict[str, Any] | None) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "system": system_blocks,
            "tools": tool_blocks,
            "messages": messages,
        }
        if thinking:
            kwargs["thinking"] = thinking

        stream_buf = ""  # content tokens awaiting a flush boundary
        think_buf = ""   # thinking tokens, flushed separately
        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                if delta.type == "text_delta":
                    stream_buf += delta.text
                    if log_fn and (
                        "\n" in stream_buf
                        or stream_buf.endswith((".", "!", "?", "…"))
                        or len(stream_buf) >= 120
                    ):
                        if stream_buf.strip():
                            log_fn("text", stream_buf.strip())
                        stream_buf = ""
                elif delta.type == "thinking_delta":
                    think_buf += delta.thinking
                    if log_fn and len(think_buf) >= 120:
                        log_fn("thinking", think_buf.strip())
                        think_buf = ""
            if log_fn:
                if think_buf.strip():
                    log_fn("thinking", think_buf.strip())
                if stream_buf.strip():
                    log_fn("text", stream_buf.strip())
            return stream.get_final_message()

    if _thinking_supported:
        try:
            return _run(_THINKING_CONFIG)
        except anthropic.BadRequestError as exc:
            if "thinking" not in str(exc).lower():
                raise
            # Adaptive thinking is rejected by pre-4.6 models. ANTHROPIC_MODEL is
            # operator-configurable, so degrade for the rest of the process
            # instead of failing every Claude build on a pinned older model.
            _thinking_supported = False
            logger.warning(
                "Model %s rejected adaptive thinking — continuing without it: %s", model, exc,
            )
    return _run(None)


