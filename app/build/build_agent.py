"""Build agent — Claude with file-system tools that implements a blueprint."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import anthropic

from app.build.agent_tools import TOOLS, execute_tool
from app.build.build_logger import tool_message

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 80
_WARN_AT_ITERATION = 50

# ---------------------------------------------------------------------------
# Template-specific directory / component scaffolding guidance
# ---------------------------------------------------------------------------

_FLUTTER_STRUCTURE = """
ARCHITECTURE: Clean Architecture with Feature-first organisation.

EXACT FOLDER STRUCTURE — call create_directory for every path below before writing any files:

  lib/core/error/
  lib/core/network/
  lib/core/usecases/
  lib/core/utils/
  lib/core/theme/

  For EACH feature extracted from the blueprint, create:
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

LAYER RESPONSIBILITIES:

  core/error/
    exceptions.dart  — AppException subclasses (NetworkException, CacheException, etc.)
    failures.dart    — Failure sealed class / subclasses for Either<Failure, T>

  core/network/
    api_client.dart  — Dio / http base client with interceptors (auth header, logging)
    network_info.dart — connectivity check (dart:io InternetAddress.lookup)

  core/usecases/
    usecase.dart     — abstract UseCase<Type, Params> interface

  core/utils/
    constants.dart   — API base URL, timeout durations, shared string keys

  core/theme/
    app_theme.dart   — ThemeData (light + dark), color palette, typography, spacing

  features/{feature}/data/datasources/
    {feature}_remote_datasource.dart  — HTTP/Firebase calls, returns Models
    {feature}_local_datasource.dart   — SharedPreferences / Hive / SQLite caching

  features/{feature}/data/models/
    {entity}_model.dart  — extends the domain Entity, adds fromJson / toJson

  features/{feature}/data/repositories/
    {feature}_repository_impl.dart  — implements domain repository, wires remote+local

  features/{feature}/domain/entities/
    {entity}.dart  — plain Dart class, no framework dependencies

  features/{feature}/domain/repositories/
    {feature}_repository.dart  — abstract repository interface

  features/{feature}/domain/usecases/
    {action}_usecase.dart  — single public call() that returns Either<Failure, T>

  features/{feature}/presentation/bloc/
    {feature}_bloc.dart   — BLoC class
    {feature}_event.dart  — sealed event classes
    {feature}_state.dart  — sealed state classes

  features/{feature}/presentation/pages/
    {feature}_page.dart   — full screen, uses BlocBuilder/BlocConsumer

  features/{feature}/presentation/widgets/
    {feature}_form.dart, {feature}_card.dart, etc. — stateless/stateful sub-widgets

ROOT FILES:
  lib/injection_container.dart — GetIt service locator registering all blocs, repos, usecases, datasources
  lib/app.dart                 — MaterialApp with theme, BlocProviders, named routes
  lib/main.dart                — runApp, WidgetsFlutterBinding, init injection_container

pubspec.yaml — add: flutter_bloc, equatable, get_it, dartz, dio, shared_preferences,
               connectivity_plus, and any feature-specific packages (firebase_*, etc.)

BUILD ORDER (strictly follow):
  1. core/ files first (error, network, utils, theme)
  2. domain/ layer for every feature (entities → repositories → usecases)
  3. data/ layer for every feature (models → datasources → repository_impl)
  4. presentation/widgets/ for every feature (reusable sub-widgets)
  5. presentation/bloc/ for every feature
  6. presentation/pages/ for every feature
  7. injection_container.dart (wire everything together)
  8. app.dart, main.dart
  9. Generate all image/video assets, declare in pubspec.yaml
  10. pubspec.yaml — finalize all dependencies
"""

_NEXT_STRUCTURE = """
ARCHITECTURE: Next.js 14 App Router — Server-side API Routes + Client Components.

RULE: Any operation that touches a database, reads secrets, or must be protected
goes in app/api/**  (Route Handlers running on the server).
Client pages/components call these API routes via fetch — they never import server
modules directly.

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
  lib/db.ts                     — database client singleton (Prisma / Supabase / mongoose)
  lib/auth.ts                   — session helpers, JWT sign/verify, cookie utilities  [AUTH ONLY]
  lib/validations.ts            — Zod schemas for validating request bodies

  ── Client utilities (safe to import in client components) ──
  lib/api.ts                    — typed fetch wrapper that calls /api/* routes
  lib/utils.ts                  — shared pure helpers (formatDate, cn, etc.)

  hooks/                        — useAuth.ts [AUTH ONLY], use{Feature}.ts — SWR / React Query hooks
  types/                        — index.ts with all shared TypeScript interfaces
  middleware.ts                 — Next.js edge middleware (protects /app/* routes)  [AUTH ONLY]
  public/images/                — AI-generated assets

API ROUTE PATTERN — use this shape for every route.ts:
  import {{ NextRequest, NextResponse }} from 'next/server'
  import {{ z }} from 'zod'
  import {{ db }} from '@/lib/db'
  import {{ getSession }} from '@/lib/auth'

  const Schema = z.object({{ ... }})

  export async function GET(req: NextRequest) {{
    const session = await getSession(req)
    if (!session) return NextResponse.json({{ error: 'Unauthorized' }}, {{ status: 401 }})
    const data = await db...
    return NextResponse.json(data)
  }}

  export async function POST(req: NextRequest) {{
    const session = await getSession(req)
    if (!session) return NextResponse.json({{ error: 'Unauthorized' }}, {{ status: 401 }})
    const body = Schema.parse(await req.json())
    const result = await db...
    return NextResponse.json(result, {{ status: 201 }})
  }}

BUILD ORDER (strictly follow; skip [AUTH ONLY] steps if auth decision is NO):
  1.  types/index.ts                        — all shared TypeScript interfaces
  2.  lib/db.ts                             — database client
  3.  lib/auth.ts                           — session / JWT helpers              [AUTH ONLY]
  4.  lib/validations.ts                    — Zod schemas (one per resource)
  5.  lib/api.ts                            — client-side fetch wrapper
  6.  lib/utils.ts                          — shared helpers
  7.  app/api/auth/login/route.ts           — POST: validate → hash → session    [AUTH ONLY]
  8.  app/api/auth/register/route.ts        — POST: validate → hash → user       [AUTH ONLY]
  9.  app/api/auth/logout/route.ts          — POST: clear session cookie          [AUTH ONLY]
  10. app/api/auth/me/route.ts              — GET: return session user             [AUTH ONLY]
  11. For each feature: app/api/{f}/route.ts and app/api/{f}/[id]/route.ts
  12. middleware.ts                         — protect (app)/* group               [AUTH ONLY]
  13. components/ui/*.tsx                  — ALL base components before any page
  14. components/layout/*.tsx              — layout components
  15. hooks/useAuth.ts                     — auth hook                            [AUTH ONLY]
      hooks/use{Feature}.ts               — SWR/RQ hooks calling lib/api.ts
  16. app/layout.tsx                       — root layout with providers
  17. app/(auth)/login/page.tsx            — login form, calls /api/auth/login    [AUTH ONLY]
  18. app/(auth)/register/page.tsx         — register form, calls /api/auth/register [AUTH ONLY]
  19. app/(app)/layout.tsx                 — authenticated shell                  [AUTH ONLY → otherwise use plain app/layout.tsx]
  20. For each feature: app/(app)/{f}/page.tsx using the components and hooks
  21. Generate all image/video assets, reference in pages
  22. tailwind.config.ts / globals.css     — theme tokens
  23. package.json                         — finalize dependencies (zod, swr, etc.)
"""

_RN_STRUCTURE = """
ARCHITECTURE: Feature-Sliced Design with Redux Toolkit.

EXACT FOLDER STRUCTURE — call create_directory for every path before writing files:

  src/app/                      — Redux store root
  For each feature from the blueprint:
    src/features/{feature}/api/
    src/features/{feature}/components/
    src/features/{feature}/screens/
    src/features/{feature}/slice/
    src/features/{feature}/types/
    src/features/{feature}/hooks/

  NOTE: if a feature is "auth" (login / register / session), create it ONLY if
  auth decision is YES. If auth decision is NO, do not create the auth feature,
  no login/register screens, no auth slice or auth API calls — skip entirely.

  src/navigation/               — navigator files
  src/services/                 — shared HTTP client
  src/hooks/                    — shared app-level hooks
  src/styles/                   — shared style constants
  src/utils/                    — constants, helpers
  assets/images/                — AI-generated assets

LAYER RESPONSIBILITIES:

  src/app/store.ts              — configureStore with all slice reducers
  src/app/rootReducer.ts        — combineReducers

  src/features/{feature}/api/{feature}Api.ts
    — RTK Query createApi endpoints or plain axios calls

  src/features/{feature}/components/{FeatureName}Form.tsx
    — reusable feature-specific UI components (no navigation logic)

  src/features/{feature}/screens/{FeatureName}Screen.tsx
    — full screen component, connects store, uses feature components

  src/features/{feature}/slice/{feature}Slice.ts
    — createSlice with actions, reducers, selectors

  src/features/{feature}/types/{feature}.types.ts
    — TypeScript interfaces for this feature's data

  src/features/{feature}/hooks/use{FeatureName}.ts
    — custom hook encapsulating slice dispatch + selectors

  src/navigation/AppNavigator.tsx   — root Stack/Tab navigator
  src/services/httpClient.ts        — axios instance with interceptors
  src/hooks/useAppDispatch.ts       — typed dispatch hook
  src/styles/tailwind.config.js     — NativeWind / StyleSheet tokens
  src/utils/constants.ts            — API_URL, storage keys, etc.
  src/App.tsx                       — Provider + NavigationContainer root

BUILD ORDER (strictly follow; skip auth feature steps if auth decision is NO):
  1. src/utils/constants.ts and src/services/httpClient.ts
  2. types files for every feature  (skip auth feature if auth decision is NO)
  3. slice files for every feature  (skip auth slice if auth decision is NO)
  4. src/app/store.ts + rootReducer.ts (import all slices)
  5. api files for every feature    (skip auth API if auth decision is NO)
  6. hooks for every feature + src/hooks/useAppDispatch.ts
  7. feature components/ (reusable, no screens yet)
  8. feature screens/               (skip login/register screens if auth decision is NO)
  9. src/navigation/AppNavigator.tsx — if auth YES: include auth stack; if NO: go straight to main stack
  10. src/App.tsx
  11. Generate all image assets, reference in screens
  12. package.json / app.json — finalize dependencies
"""

_STRUCTURE_MAP = {
    "flutter": _FLUTTER_STRUCTURE,
    "next": _NEXT_STRUCTURE,
    "react_native": _RN_STRUCTURE,
}

_BUILD_PREAMBLE = """You are the Forgefy Build Agent.
Your task: implement a complete, working application from the blueprint by writing files in the workspace.

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

PHASE 1 · Explore
  • list_files on '.' to see the existing template
  • Read key config files to understand conventions

PHASE 2 · Scaffold directories
  • Call create_directory for every folder listed in the structure below
  • Before creating each file, write one short sentence narrating what you are doing
    (e.g. "Creating LoginScreen…", "Building AuthService…")
    This sentence appears in the user's live build log.

PHASE 3 · Reusable components / widgets  ← DO THIS BEFORE SCREENS
  • Build all shared UI components first so screens can import them
  • Keep each component focused on one responsibility

PHASE 4 · Models & services
  • Data models with serialisation (fromJson/toJson, TypeScript interfaces)
  • Service classes: one per domain (auth, API, local storage, etc.)
  • State management wired up

PHASE 5 · Screens / pages
  • One file per screen/route listed in the blueprint
  • Every screen must use the reusable components from Phase 3

PHASE 6 · Assets
  • Use generate_image for every visual element: backgrounds, hero images,
    onboarding artwork, icons, illustrations — do NOT leave image slots empty
  • Use generate_video for splash/onboarding animations where appropriate
  • After generating an asset, immediately reference it in code:
      Flutter    → Image.asset('assets/images/<file>') + declare in pubspec.yaml
      Next.js    → <img src="/images/<file>"> or next/image
      React Native → require('./assets/images/<file>')

PHASE 7 · Configuration
  • Update the app name everywhere (pubspec.yaml, package.json, app.json, Info.plist, etc.)
  • Add all required third-party dependencies
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
• Implement EVERY feature listed in the blueprint — nothing optional
• Handle loading states, empty states, and basic error states in every screen
• Add input validation where the app collects user data
• Style the app consistently using the color/theme constants you define
• Do not leave any generated image/video slot with a placeholder URL
"""


def _build_system(template_key: str) -> str:
    structure = _STRUCTURE_MAP.get(template_key, _NEXT_STRUCTURE)
    return _BUILD_PREAMBLE + structure + _BUILD_SUFFIX


# ---------------------------------------------------------------------------
# Update agent system prompt
# ---------------------------------------------------------------------------
_UPDATE_SYSTEM = """You are the Forgefy Update Agent making targeted changes to an existing application.

MANDATORY FIRST STEP: Call list_files('.') to understand the project structure before doing anything else.

CRITICAL RULES — you MUST follow these or the task fails:
1. ALWAYS use the write_file tool to make changes. You MUST call write_file at least once — describing changes in text is NOT acceptable.
2. Always start by calling list_files('.') then read the relevant source files before writing anything.
3. For broad requests, break them into specific sub-tasks and implement each one:
   - "add onboarding screen" → list files, find navigator/router, create the onboarding screen file, update the navigator to include it
   - "add animations" → list files, identify which screens to animate, rewrite each with animation code
   - "add dark mode" → find theme file, update colors, add toggle logic
4. If the change requires a new screen/route, also update the navigator/router file.
5. Narrate each step briefly before each tool call: "Reading navigator…", "Writing OnboardingScreen…"
6. After ALL write_file calls are done, write a summary starting with DONE: describing exactly what changed.

IMPORTANT:
- A response of "." or any single character is WRONG — always implement the full change with write_file.
- Never say DONE without having called write_file at least once.
- If the request is vague, make a reasonable implementation — do not refuse or return empty."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_build_agent(
    workspace: Path,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, int]:
    """Run the build agent tool loop; return (summary, total_tokens_used)."""
    client = anthropic.Anthropic(api_key=api_key)
    system = _build_system(template_key)
    user_msg = (
        f"App name: {app_name}\n"
        f"Template: {template_key}\n\n"
        f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
        "Build this application now following the phases in your instructions. "
        "Narrate each step as you go. When finished, write a user-friendly summary starting with DONE: that describes what was built — screens, features, and anything notable."
    )
    return _loop(client, model, system, workspace, user_msg, log_fn)


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


def _ollama_loop(
    base_url: str,
    model: str,
    system: str,
    workspace: Path,
    initial_user_msg: str,
    timeout: int = 300,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, int]:
    """Ollama tool-use agent loop. Returns (summary, 0) — Ollama doesn't expose token counts."""
    import requests as _req

    # Keep tool results short so the context doesn't balloon across iterations.
    _TOOL_RESULT_LIMIT = 1500
    # Sliding window: system + first user msg are always kept; only the last N
    # assistant/tool pairs are retained so the context stays within num_ctx.
    _HISTORY_PAIRS = 6  # = 12 messages max in the rolling window

    url = f"{base_url.rstrip('/')}/api/chat"
    ollama_tools = _tools_to_ollama_format(TOOLS)
    # Slot 0 = system, slot 1 = initial user task — never dropped.
    anchor: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": initial_user_msg},
    ]
    history: list[dict[str, Any]] = []  # assistant + tool messages, pruned each turn
    last_text = ""
    write_calls = 0  # track whether the agent actually wrote any files

    def _trimmed_messages() -> list[dict[str, Any]]:
        """Return anchor + the last _HISTORY_PAIRS*2 history messages."""
        return anchor + history[-(_HISTORY_PAIRS * 2):]

    for iteration in range(_MAX_ITERATIONS):
        if iteration == _WARN_AT_ITERATION and log_fn:
            log_fn("warning", f"Build is complex ({iteration} steps so far) — finishing up…")

        try:
            # stream=True: timeout applies per-chunk, not for the full response,
            # so long generations don't hit the read timeout.
            with _req.post(
                url,
                json={
                    "model": model,
                    "messages": _trimmed_messages(),
                    "tools": ollama_tools,
                    "stream": True,
                    "options": {
                        "num_ctx": 8192,
                        "num_predict": 4096,
                    },
                },
                timeout=(30, None),
                stream=True,
            ) as resp:
                resp.raise_for_status()
                content_parts: list[str] = []
                tool_calls: list[dict] = []
                stream_buf = ""       # accumulate content tokens until a flush boundary
                think_buf = ""        # accumulate thinking tokens separately
                in_think_tag = False  # track inline <think> blocks in content
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    chunk = json.loads(raw_line)
                    msg = chunk.get("message", {})

                    # ── Thinking tokens (Ollama thinking-model field) ──────────
                    think_token = msg.get("thinking") or ""
                    if think_token:
                        think_buf += think_token
                        if log_fn and len(think_buf) >= 120:
                            log_fn("thinking", think_buf.strip())
                            think_buf = ""

                    # ── Content tokens ─────────────────────────────────────────
                    token = msg.get("content") or ""
                    if token:
                        # Route inline <think>…</think> to the thinking buffer
                        # so they don't pollute the content sent back to the model.
                        if "<think>" in token:
                            in_think_tag = True
                        if in_think_tag:
                            think_buf += token
                            if log_fn and len(think_buf) >= 120:
                                log_fn("thinking", think_buf.strip())
                                think_buf = ""
                            if "</think>" in token:
                                in_think_tag = False
                                think_buf = ""  # done with this think block
                        else:
                            content_parts.append(token)
                            stream_buf += token
                            # Flush to log at natural sentence boundaries or 120 chars
                            if log_fn and (
                                "\n" in stream_buf
                                or stream_buf.endswith((".", "!", "?", "…"))
                                or len(stream_buf) >= 120
                            ):
                                log_fn("text", stream_buf.strip())
                                stream_buf = ""

                    if chunk.get("done"):
                        tool_calls = msg.get("tool_calls") or []
                        if log_fn:
                            if think_buf.strip():
                                log_fn("thinking", think_buf.strip())
                                think_buf = ""
                            if stream_buf.strip():
                                log_fn("text", stream_buf.strip())
                                stream_buf = ""

                content_text: str = "".join(content_parts)
        except Exception as exc:
            raise RuntimeError(f"Ollama build agent request failed: {exc}") from exc

        history.append({
            "role": "assistant",
            "content": content_text,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        })

        if content_text:
            last_text = content_text
            if "DONE" in content_text.upper():
                if write_calls == 0:
                    # Agent claimed done without writing anything — push back once
                    if log_fn:
                        log_fn("info", "Agent said DONE without writing files — asking it to implement…")
                    history.append({"role": "assistant", "content": content_text})
                    history.append({
                        "role": "user",
                        "content": (
                            "You said you were done but you haven't called write_file yet. "
                            "Please implement the changes now using write_file. "
                            "Do not just describe what to do — actually write the code."
                        ),
                    })
                    continue
                # Don't emit done here — update_worker will use the returned summary
                return content_text, 0

        if not tool_calls:
            if write_calls == 0 and last_text:
                # No tool calls and no writes — push back once to get actual file output
                if log_fn:
                    log_fn("info", "No files written yet — asking agent to write the code…")
                history.append({"role": "assistant", "content": last_text})
                history.append({
                    "role": "user",
                    "content": (
                        "You haven't written any files yet. "
                        "Use the write_file tool to implement the changes now."
                    ),
                })
                continue
            # Don't emit done here — update_worker will use the returned summary
            return last_text or "Done.", 0

        for call in tool_calls:
            func = call.get("function", {})
            tool_name = func.get("name", "")
            tool_input = func.get("arguments", {})
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    tool_input = {}
            if tool_name == "write_file":
                write_calls += 1
            if log_fn:
                log_fn("tool", tool_message(tool_name, tool_input))
            result = execute_tool(tool_name, tool_input, workspace)
            # Truncate large results (e.g. read_file on a big file) so they
            # don't blow up the context window on the next iteration.
            if len(result) > _TOOL_RESULT_LIMIT:
                result = result[:_TOOL_RESULT_LIMIT] + "\n…[truncated]"
            history.append({"role": "tool", "content": result})

    if log_fn:
        log_fn("error", "Agent reached iteration limit.")
    return "Agent reached iteration limit.", 0


def run_build_agent_ollama(
    workspace: Path,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    base_url: str,
    model: str,
    timeout: int = 300,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, int]:
    """Run the build agent using local Ollama with tool calls; return (summary, 0)."""
    system = _build_system(template_key)
    user_msg = (
        f"App name: {app_name}\n"
        f"Template: {template_key}\n\n"
        f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
        "Build this application now following the phases in your instructions. "
        "Narrate each step as you go. When finished, write a user-friendly summary starting with DONE: that describes what was built — screens, features, and anything notable."
    )
    return _ollama_loop(base_url, model, system, workspace, user_msg, timeout, log_fn)


def run_update_agent(
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, int]:
    """Run the update agent; return (summary, total_tokens_used)."""
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = (
        f"App name: {app_name}\n"
        f"Existing blueprint context:\n{json.dumps(blueprint, indent=2)}\n\n"
        f"User's update request: {prompt}\n\n"
        "Apply this change now. Read the relevant files first, make the changes, "
        "then write a user-friendly summary starting with DONE: that describes exactly what was changed."
    )
    return _loop(client, model, _UPDATE_SYSTEM, workspace, user_msg, log_fn)


def run_update_agent_ollama(
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    base_url: str,
    model: str,
    timeout: int = 300,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, int]:
    """Run the update agent using local Ollama; return (summary, 0)."""
    user_msg = (
        f"App name: {app_name}\n"
        f"Existing blueprint context:\n{json.dumps(blueprint, indent=2)}\n\n"
        f"User's update request: {prompt}\n\n"
        "Apply this change now. Read the relevant files first, make the changes, "
        "then write a user-friendly summary starting with DONE: that describes exactly what was changed."
    )
    return _ollama_loop(base_url, model, _UPDATE_SYSTEM, workspace, user_msg, timeout, log_fn)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def _loop(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    workspace: Path,
    initial_user_msg: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, int]:
    """Agent tool loop. Returns (summary, total_tokens_used)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_user_msg}]
    total_tokens = 0
    write_calls = 0
    pushback_sent = False

    for iteration in range(_MAX_ITERATIONS):
        if iteration == _WARN_AT_ITERATION and log_fn:
            log_fn("warning", f"Build is complex ({iteration} steps so far) — finishing up…")

        response = client.messages.create(
            model=model,
            max_tokens=8096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        total_tokens += response.usage.input_tokens + response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict[str, Any]] = []
        last_text = ""
        done_text: str | None = None

        for block in response.content:
            if block.type == "text":
                last_text = block.text
                logger.debug("Agent text: %s", block.text[:120])
                if log_fn and block.text.strip():
                    preview = block.text.strip()[:160]
                    if len(block.text.strip()) > 160:
                        preview += "…"
                    log_fn("text", preview)
                if "DONE" in block.text.upper():
                    done_text = block.text

            elif block.type == "tool_use":
                if block.name == "write_file":
                    write_calls += 1
                msg = tool_message(block.name, block.input)
                logger.debug("Tool %s → %s", block.name, msg)
                if log_fn:
                    log_fn("tool", msg)
                result = execute_tool(block.name, block.input, workspace)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

        # If agent said DONE but hasn't written any files, push back once
        if done_text is not None and not tool_results:
            if write_calls == 0 and not pushback_sent:
                pushback_sent = True
                if log_fn:
                    log_fn("info", "Agent said DONE without writing files — asking it to implement…")
                messages.append({
                    "role": "user",
                    "content": (
                        "You said you were done but you haven't called write_file yet. "
                        "Please implement the changes now using write_file. "
                        "Start with list_files('.') to explore the project, then write the code."
                    ),
                })
                continue
            return done_text, total_tokens

        if response.stop_reason == "end_turn":
            if write_calls == 0 and not pushback_sent:
                # Agent stopped without writing any files — push back once
                pushback_sent = True
                if log_fn:
                    log_fn("info", "No files written yet — asking agent to write the code…")
                messages.append({
                    "role": "user",
                    "content": (
                        "You haven't written any files yet. "
                        "Use the write_file tool to implement the changes now. "
                        "Start by calling list_files('.') to see the project structure, "
                        "then read the relevant files and write the implementation."
                    ),
                })
                continue
            return last_text or "Done.", total_tokens

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    if log_fn:
        log_fn("error", "Agent reached iteration limit.")
    return "Agent reached iteration limit.", total_tokens
