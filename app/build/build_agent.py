"""Build agent — Claude with file-system tools that implements a blueprint."""
from __future__ import annotations

import json
import logging
import re
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

ROOT FILES (mandatory — do NOT rename these):
  lib/injection_container.dart  — GetIt registering all blocs, repos, usecases, datasources
  lib/app.dart                  — MaterialApp with theme, BlocProviders, named routes
  lib/main.dart                 — runApp, WidgetsFlutterBinding, init injection_container

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

CANONICAL FILE NAMES — use these exact names, no variations:

  src/app/store.ts                                   — configureStore with all slice reducers
  src/app/rootReducer.ts                             — combineReducers
  src/features/{feature}/api/{feature}Api.ts         — RTK Query or axios calls
  src/features/{feature}/components/{Feature}Form.tsx — reusable UI (no navigation logic)
  src/features/{feature}/screens/{Feature}Screen.tsx  — full screen, connects store
  src/features/{feature}/slice/{feature}Slice.ts     — createSlice actions/reducers/selectors
  src/features/{feature}/types/{feature}.types.ts    — TypeScript interfaces for this feature
  src/features/{feature}/hooks/use{Feature}.ts       — hook encapsulating dispatch + selectors
  src/navigation/AppNavigator.tsx                    — root Stack/Tab navigator (DO NOT rename)
  src/services/httpClient.ts                         — axios instance with interceptors
  src/hooks/useAppDispatch.ts                        — typed dispatch hook
  src/styles/tailwind.config.js                      — NativeWind / StyleSheet tokens
  src/utils/constants.ts                             — API_URL, storage keys, etc.
  src/App.tsx                                        — Provider + NavigationContainer root

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

  Root (mandatory, do NOT rename):
  lib/injection_container.dart   lib/app.dart   lib/main.dart

RULES:
  • Pages go in presentation/pages/ ONLY. Never at lib/ root or lib/screens/.
  • Widgets go in presentation/widgets/ ONLY.
  • Data models go in data/models/ ONLY.
  • If you add a new feature, create ALL sub-folders listed above.
  • If you add a new screen, register it in lib/app.dart (GoRouter or named routes).
""",
    "react_native": """\
REACT NATIVE FOLDER STRUCTURE — every file you create MUST follow this layout:

  src/app/store.ts               src/app/rootReducer.ts
  src/navigation/AppNavigator.tsx   ← only navigator file, do NOT rename
  src/services/httpClient.ts
  src/hooks/useAppDispatch.ts
  src/styles/tailwind.config.js
  src/utils/constants.ts
  src/App.tsx                       ← root entry, do NOT rename

  For each feature → ALL sub-folders are mandatory:
  src/features/{feature}/api/{feature}Api.ts
  src/features/{feature}/components/{Feature}Form.tsx   (or Card, List, etc.)
  src/features/{feature}/screens/{Feature}Screen.tsx
  src/features/{feature}/slice/{feature}Slice.ts
  src/features/{feature}/types/{feature}.types.ts
  src/features/{feature}/hooks/use{Feature}.ts

RULES:
  • Screens go in features/{feature}/screens/ ONLY. Never in src/screens/ or root.
  • Components go in features/{feature}/components/ ONLY.
  • Slice files go in features/{feature}/slice/ ONLY.
  • If you add a new screen, register it in src/navigation/AppNavigator.tsx.
  • If you add a new slice, add it to src/app/store.ts and rootReducer.ts.
""",
    "next": """\
NEXT.JS FOLDER STRUCTURE — every file you create MUST follow this layout:

  app/api/{feature}/route.ts          — server-side: GET (list) + POST (create)
  app/api/{feature}/[id]/route.ts     — server-side: GET + PUT + DELETE
  app/(app)/{feature}/page.tsx        — client page for this feature
  components/{feature}/               — feature-specific reusable components
  components/ui/                      — shared primitive components
  components/layout/                  — Header, Sidebar, Footer, etc.
  lib/db.ts   lib/auth.ts   lib/validations.ts   lib/api.ts   lib/utils.ts
  hooks/use{Feature}.ts
  types/index.ts
  middleware.ts                        — route protection [auth only]

RULES:
  • API logic goes in app/api/ ONLY — never in page.tsx files.
  • New pages go in app/(app)/{feature}/page.tsx.
  • Add a nav link in components/layout/ when adding a new page.
  • All shared types in types/index.ts — do not scatter them across files.

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

REUSE CORE COMPONENTS FIRST:
Before creating any UI widget/component, check if one already exists in
lib/core/widgets/ (Flutter), src/core/components/ (RN), or components/ui/ (Next.js).
Reuse and extend it — do NOT duplicate.

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

PHASE 0 (auth) — if auth IS needed: the login/register screens MUST use the pre-built
  AppTextField and AppButton core components. No custom text fields. Include a branded
  header using the display_font. Password field has show/hide toggle.

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
  • Core components are PRE-BUILT in lib/core/widgets/ (Flutter),
    src/core/components/ (RN), or components/ui/ (Next.js).
  • Only write FEATURE-SPECIFIC components here. Each must import from core — never
    redefine base styles.
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
• Implement EVERY feature listed in the blueprint — nothing optional
• Handle loading states, empty states, and basic error states in every screen
• Add input validation where the app collects user data
• Style the app consistently using the color/theme constants you define
• Do not leave any generated image/video slot with a placeholder URL
• ALWAYS create THREE environment files:

  1. .env  ← committed to GitHub
     Contains ONLY public-safe variables with placeholder values. Public-safe means:
       — NEXT_PUBLIC_* (bundled into browser JS — cannot be secret by design)
       — Supabase URL + anon key (security enforced by Row Level Security, not the key)
       — Firebase client config: apiKey, authDomain, projectId, storageBucket,
         messagingSenderId, appId (security enforced by Firebase Security Rules)
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


def _build_system(template_key: str) -> str:
    structure = _STRUCTURE_MAP.get(template_key, _NEXT_STRUCTURE)
    return _DESIGN_MANDATE + _BUILD_PREAMBLE + structure + _BUILD_SUFFIX


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

REUSE CORE COMPONENTS — check before creating anything new:
  lib/core/widgets/ (Flutter) | src/core/components/ (RN) | components/ui/ (Next.js)
  If a matching component exists there, use it — do not create a duplicate.

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
- Narrate briefly before each tool call: "Reading AppNavigator…", "Writing OnboardingPage…"
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
  Register route in lib/app.dart.

Animations:
  Entrance: AnimatedOpacity + SlideTransition triggered in initState via AnimationController.
  List stagger: wrap each item in AnimationBuilder with staggered begin values.
  Page transitions: use PageRouteBuilder with custom transitionsBuilder.
  Hero: wrap shared elements in Hero widget with matching tag.

Dark mode:
  Add ThemeMode state to MaterialApp. Toggle via a provider/bloc. Persist in SharedPreferences.

New screen:
  Create in lib/features/{feature}/presentation/pages/.
  Add named route to lib/app.dart routes map or GoRouter.

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
  File: src/features/onboarding/screens/OnboardingScreen.tsx
  Use FlatList or ScrollView with pagingEnabled. Store completion in AsyncStorage.
  Add to AppNavigator.tsx as the initial screen when not completed.

Animations:
  Entrance: Animated.timing with useRef(new Animated.Value(0)).
  List stagger: run Animated.stagger on mount.
  Layout: LayoutAnimation.configureNext before state changes.

New screen:
  Create in src/features/{feature}/screens/.
  Add to Stack.Navigator in src/navigation/AppNavigator.tsx.

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

RULE 6 — REACT NATIVE COMMON MISTAKES:
  ✗  import { View } from 'react-native-web'  → ✓  from 'react-native'
  ✗  StyleSheet.create({ x: { color: '#fff' } }) → ✓  use tokens: colors.xxx
  ✗  navigation.navigate('Screen', params)  → type must match the Navigator's param list
  ✗  AsyncStorage from 'react-native'  → ✓  from '@react-native-async-storage/async-storage'"""


# ---------------------------------------------------------------------------
# Planner — single text-only API call that returns a structured execution plan
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """You are a software architect. Your ONLY job is to analyse a change request and produce a precise execution plan.

You do NOT write code. You do NOT call tools. You do NOT modify files.

Return ONLY valid JSON — absolutely no other text before or after:
{
  "summary": "<one sentence — what will be built or changed>",
  "skip_design_agent": false,
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
        file_tree = execute_tool("list_files", {"path": "."}, workspace)
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
        f"Current workspace files:\n{file_tree[:1200]}\n\n"
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

        else:  # Qwen3 / Ollama
            import requests as _req
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

PASS 3 — Core component reuse check
Search for any custom button, text field, or card implementation that duplicates a
core component in lib/core/widgets/ / src/core/components/ / components/ui/.
If duplicates exist: merge into the canonical path, delete_file the duplicate.

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
   - Flutter: new screens must be registered in lib/app.dart (GoRouter or named routes)
   - Next.js: new pages must have a link in components/layout/; new packages in package.json
   - React Native: new screens must be in src/navigation/AppNavigator.tsx
5. ── STRUCTURE CHECK (see FOLDER STRUCTURE RULES in your task) ──
   a. For Flutter: verify every new feature has ALL required sub-folders:
      data/datasources/, data/models/, data/repositories/,
      domain/entities/, domain/repositories/, domain/usecases/,
      presentation/bloc/, presentation/pages/, presentation/widgets/
      And that lib/injection_container.dart, lib/app.dart, lib/main.dart exist.
   b. For React Native: verify every new feature has ALL required sub-folders:
      api/, components/, screens/, slice/, types/, hooks/
      And that src/App.tsx, src/navigation/AppNavigator.tsx exist.
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

    structure_rules = _UPDATE_STRUCTURE_RULES.get(template_key, "")
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
    cancel_fn: Callable[[], bool] | None = None,
    max_iterations: int = _MAX_ITERATIONS,
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

    for iteration in range(max_iterations):
        if cancel_fn and cancel_fn():
            if log_fn:
                log_fn("warning", "Agent stopped by user.")
            return "Stopped by user.", 0
        warn_at = max(1, max_iterations - 10)
        if iteration == warn_at and log_fn:
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
                timeout=(30, timeout),
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
                    # assistant message already appended above — only add the user pushback
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
                    log_fn("file_written", tool_input.get("path", ""))
            if log_fn:
                log_fn("tool", tool_message(tool_name, tool_input))
            result = execute_tool(tool_name, tool_input, workspace, log_fn)
            # Truncate large results (e.g. read_file on a big file) so they
            # don't blow up the context window on the next iteration.
            if len(result) > _TOOL_RESULT_LIMIT:
                result = result[:_TOOL_RESULT_LIMIT] + "\n…[truncated]"
            history.append({"role": "tool", "content": result})

    if log_fn:
        log_fn("warning", "Agent reached iteration limit.")
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
    structure_rules = _UPDATE_STRUCTURE_RULES.get(template_key, "")
    structure_section = (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "MANDATORY FOLDER STRUCTURE — all new files must follow this\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + structure_rules
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    ) if structure_rules else ""

    file_listing = ""
    if workspace is not None:
        scan = _scan_workspace(workspace)
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
            "EXISTING PROJECT FILES — read before creating anything\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Before calling write_file for any new file:\n"
            "  1. Search this list for any file with the SAME NAME.\n"
            "  2. Search for any file that serves the SAME PURPOSE\n"
            "     (e.g. auth_service.dart ≈ authentication_service.dart,\n"
            "      home_screen.dart ≈ home_page.dart, userApi.ts ≈ user_service.ts).\n"
            "  3. If found → read it first, then WRITE YOUR CHANGES TO THAT EXISTING PATH.\n"
            "  4. Only create a brand-new path if you confirm nothing similar already exists.\n"
            "  NEVER create a second file that does the same job as an existing one.\n\n"
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
) -> str:
    """Like _update_user_msg but prepends the design agent's output so the executor knows
    exactly which components and tokens were prepared."""
    base = _update_user_msg(app_name, template_key, blueprint, prompt, plan, workspace)
    if not design_brief:
        return base
    brief_block = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "DESIGN SYSTEM BRIEF — use these components and tokens\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + design_brief.strip() + "\n"
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    return brief_block + base


# Per-agent iteration budgets — focused agents get fewer iterations to save tokens
_ITERS_DESIGN = 30
_ITERS_EXEC = 50
_ITERS_VALIDATE = 25
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


def _should_skip_security(plan: dict[str, Any] | None) -> bool:
    if not plan:
        return False
    return bool(plan.get("skip_security_agent"))


def _validation_needs_fix_pass(val_summary: str) -> bool:
    """Return True when the validator did not emit a clean VALIDATED: report.

    A clean report means the validator finished its workflow and signed off with
    a VALIDATED: line. Anything else (role explanation, timeout, error dump) means
    the executor should get a targeted fix pass with the validator's findings.
    """
    vs = val_summary.lower().strip()
    if "all checks passed" in vs:
        return False
    if vs.startswith("validated:") or "\nvalidated:" in vs:
        return False
    return True


def _make_fix_prompt(val_summary: str) -> str:
    return (
        "The validator reviewed the implementation and found unresolved issues.\n\n"
        f"Validator findings:\n{val_summary}\n\n"
        "Fix every problem described above. Read each affected file first, then apply "
        "the minimum change needed. Do not re-explain issues — just fix them."
    )


def run_update_agent(
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Plan → design? → execute → validate → security? using Claude/Anthropic."""
    if log_fn:
        log_fn("thinking", "Planning changes…")
    plan = _call_planner(
        _build_planner_msg(prompt, blueprint, workspace, template_key, app_name),
        backend="claude", api_key=api_key, model=model,
    )
    _log_plan(plan, log_fn)
    client = anthropic.Anthropic(api_key=api_key)
    total_tokens = 0

    if cancel_fn and cancel_fn():
        return "Stopped by user.", 0

    design_summary = ""
    if not _should_skip_design(plan):
        if log_fn:
            log_fn("info", "━━━ Design system phase ━━━")
            log_fn("thinking", "Preparing design system…")
        design_summary, design_tokens = _loop(
            client, model, _DESIGN_AGENT_SYSTEM, workspace,
            _design_agent_user_msg(app_name, template_key, plan, prompt, workspace),
            log_fn, cancel_fn, max_iterations=_ITERS_DESIGN,
        )
        total_tokens += design_tokens
        if cancel_fn and cancel_fn():
            return design_summary or "Stopped by user.", total_tokens
    elif log_fn:
        log_fn("info", "Design phase skipped — no new components or theme changes.")

    if log_fn:
        log_fn("info", "━━━ Execution phase ━━━")
    exec_summary, exec_tokens = _loop(
        client, model, _UPDATE_SYSTEM, workspace,
        _update_user_msg_with_brief(app_name, template_key, blueprint, prompt, plan, workspace, design_summary),
        log_fn, cancel_fn, max_iterations=_ITERS_EXEC,
    )
    total_tokens += exec_tokens
    if cancel_fn and cancel_fn():
        return exec_summary or "Stopped by user.", total_tokens

    if log_fn:
        log_fn("info", "━━━ Validation phase ━━━")
        log_fn("thinking", "Validating implementation…")
    val_summary, val_tokens = _loop(
        client, model, _VALIDATOR_SYSTEM, workspace,
        _validator_user_msg(app_name, template_key, plan, prompt, workspace, exec_summary),
        log_fn, cancel_fn, max_iterations=_ITERS_VALIDATE,
    )
    total_tokens += val_tokens
    if cancel_fn and cancel_fn():
        return val_summary or exec_summary, total_tokens

    if _validation_needs_fix_pass(val_summary):
        if log_fn:
            log_fn("info", "━━━ Post-validation fix pass ━━━")
            log_fn("thinking", "Validator found unresolved issues — running targeted fix pass…")
        fix_summary, fix_tokens = _loop(
            client, model, _UPDATE_SYSTEM, workspace,
            _make_fix_prompt(val_summary), log_fn, cancel_fn, max_iterations=_ITERS_FIX,
        )
        total_tokens += fix_tokens
        val_summary = fix_summary
        if cancel_fn and cancel_fn():
            return val_summary or exec_summary, total_tokens

    if not _should_skip_security(plan):
        if log_fn:
            log_fn("info", "━━━ Security review phase ━━━")
            log_fn("thinking", "Reviewing for security issues…")
        sec_summary, sec_tokens = _loop(
            client, model, _SECURITY_SYSTEM, workspace,
            _security_user_msg(app_name, template_key, plan, prompt, workspace),
            log_fn, cancel_fn, max_iterations=_ITERS_SECURITY,
        )
        total_tokens += sec_tokens
    elif log_fn:
        log_fn("info", "Security phase skipped — no auth/API/storage changes.")

    return val_summary or exec_summary, total_tokens


def run_update_agent_ollama(
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    base_url: str,
    model: str,
    timeout: int = 300,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Plan → design? → execute → validate → security? using Ollama/Qwen3."""
    if log_fn:
        log_fn("thinking", "Planning changes…")
    plan = _call_planner(
        _build_planner_msg(prompt, blueprint, workspace, template_key, app_name),
        backend="Qwen3", base_url=base_url, ollama_model=model, ollama_timeout=timeout,
    )
    _log_plan(plan, log_fn)
    total_tokens = 0

    if cancel_fn and cancel_fn():
        return "Stopped by user.", 0

    design_summary = ""
    if not _should_skip_design(plan):
        if log_fn:
            log_fn("info", "━━━ Design system phase ━━━")
            log_fn("thinking", "Preparing design system…")
        design_summary, design_tokens = _ollama_loop(
            base_url, model, _DESIGN_AGENT_SYSTEM, workspace,
            _design_agent_user_msg(app_name, template_key, plan, prompt, workspace),
            timeout, log_fn, cancel_fn, max_iterations=_ITERS_DESIGN,
        )
        total_tokens += design_tokens
        if cancel_fn and cancel_fn():
            return design_summary or "Stopped by user.", total_tokens
    elif log_fn:
        log_fn("info", "Design phase skipped — no new components or theme changes.")

    if log_fn:
        log_fn("info", "━━━ Execution phase ━━━")
    exec_summary, exec_tokens = _ollama_loop(
        base_url, model, _UPDATE_SYSTEM, workspace,
        _update_user_msg_with_brief(app_name, template_key, blueprint, prompt, plan, workspace, design_summary),
        timeout, log_fn, cancel_fn, max_iterations=_ITERS_EXEC,
    )
    total_tokens += exec_tokens
    if cancel_fn and cancel_fn():
        return exec_summary or "Stopped by user.", total_tokens

    if log_fn:
        log_fn("info", "━━━ Validation phase ━━━")
        log_fn("thinking", "Validating implementation…")
    val_summary, val_tokens = _ollama_loop(
        base_url, model, _VALIDATOR_SYSTEM, workspace,
        _validator_user_msg(app_name, template_key, plan, prompt, workspace, exec_summary),
        timeout, log_fn, cancel_fn, max_iterations=_ITERS_VALIDATE,
    )
    total_tokens += val_tokens
    if cancel_fn and cancel_fn():
        return val_summary or exec_summary, total_tokens

    if _validation_needs_fix_pass(val_summary):
        if log_fn:
            log_fn("info", "━━━ Post-validation fix pass ━━━")
            log_fn("thinking", "Validator found unresolved issues — running targeted fix pass…")
        fix_summary, fix_tokens = _ollama_loop(
            base_url, model, _UPDATE_SYSTEM, workspace,
            _make_fix_prompt(val_summary), timeout, log_fn, cancel_fn, max_iterations=_ITERS_FIX,
        )
        total_tokens += fix_tokens
        val_summary = fix_summary
        if cancel_fn and cancel_fn():
            return val_summary or exec_summary, total_tokens

    if not _should_skip_security(plan):
        if log_fn:
            log_fn("info", "━━━ Security review phase ━━━")
            log_fn("thinking", "Reviewing for security issues…")
        sec_summary, sec_tokens = _ollama_loop(
            base_url, model, _SECURITY_SYSTEM, workspace,
            _security_user_msg(app_name, template_key, plan, prompt, workspace),
            timeout, log_fn, cancel_fn, max_iterations=_ITERS_SECURITY,
        )
        total_tokens += sec_tokens
    elif log_fn:
        log_fn("info", "Security phase skipped — no auth/API/storage changes.")

    return val_summary or exec_summary, total_tokens


# ---------------------------------------------------------------------------
# Gemini agent loop
# ---------------------------------------------------------------------------

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_TOOL_RESULT_LIMIT = 1500


def _tools_to_gemini_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"functionDeclarations": [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }
        for t in tools
    ]}]


def _gemini_loop(
    api_key: str,
    model: str,
    system: str,
    workspace: Path,
    initial_user_msg: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
    max_iterations: int = _MAX_ITERATIONS,
) -> tuple[str, int]:
    """Gemini tool-use agent loop via REST API. Returns (summary, 0)."""
    import requests as _req

    url = _GEMINI_URL.format(model=model)
    gemini_tools = _tools_to_gemini_format(TOOLS)
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": initial_user_msg}]}]
    last_text = ""
    write_calls = 0
    pushback_sent = False

    for iteration in range(max_iterations):
        if cancel_fn and cancel_fn():
            if log_fn:
                log_fn("warning", "Agent stopped by user.")
            return "Stopped by user.", 0
        warn_at = max(1, max_iterations - 10)
        if iteration == warn_at and log_fn:
            log_fn("warning", f"Build is complex ({iteration} steps so far) — finishing up…")

        try:
            resp = _req.post(
                url,
                params={"key": api_key},
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": contents,
                    "tools": gemini_tools,
                    "generationConfig": {"maxOutputTokens": 8192},
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Gemini agent request failed: {exc}") from exc

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        finish_reason = candidate.get("finishReason", "")
        contents.append({"role": "model", "parts": parts})

        tool_calls: list[dict] = []
        done_text: str | None = None

        for part in parts:
            if "text" in part:
                txt = part["text"]
                last_text = txt
                if log_fn and txt.strip():
                    log_fn("text", txt.strip()[:160])
                if "DONE" in txt.upper():
                    done_text = txt
            elif "functionCall" in part:
                tool_calls.append(part["functionCall"])

        if done_text is not None and not tool_calls:
            if write_calls == 0 and not pushback_sent:
                pushback_sent = True
                if log_fn:
                    log_fn("info", "Agent said DONE without writing files — asking it to implement…")
                contents.append({"role": "user", "parts": [{"text": (
                    "You said you were done but haven't called write_file yet. "
                    "Implement the changes now using write_file."
                )}]})
                continue
            return done_text, 0

        if finish_reason in ("STOP", "MAX_TOKENS") and not tool_calls:
            if write_calls == 0 and not pushback_sent:
                pushback_sent = True
                if log_fn:
                    log_fn("info", "No files written yet — asking agent to write the code…")
                contents.append({"role": "user", "parts": [{"text": (
                    "You haven't written any files yet. Use write_file to implement the changes."
                )}]})
                continue
            return last_text or "Done.", 0

        tool_responses: list[dict] = []
        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_input = call.get("args", {})
            if tool_name == "write_file":
                write_calls += 1
                if log_fn:
                    log_fn("file_written", tool_input.get("path", ""))
            if log_fn:
                log_fn("tool", tool_message(tool_name, tool_input))
            result = execute_tool(tool_name, tool_input, workspace, log_fn)
            if len(result) > _TOOL_RESULT_LIMIT:
                result = result[:_TOOL_RESULT_LIMIT] + "\n…[truncated]"
            tool_responses.append({"functionResponse": {"name": tool_name, "response": {"output": result}}})

        if tool_responses:
            contents.append({"role": "user", "parts": tool_responses})

    if log_fn:
        log_fn("warning", "Agent reached iteration limit.")
    return "Agent reached iteration limit.", 0


def run_build_agent_gemini(
    workspace: Path,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, int]:
    system = _build_system(template_key)
    user_msg = (
        f"App name: {app_name}\nTemplate: {template_key}\n\n"
        f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
        "Build this application now following the phases in your instructions. "
        "Narrate each step. When finished, write DONE: <summary of what was built>."
    )
    return _gemini_loop(api_key, model, system, workspace, user_msg, log_fn)


def run_update_agent_gemini(
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Plan → design? → execute → validate → security? using Gemini."""
    if log_fn:
        log_fn("thinking", "Planning changes…")
    plan = _call_planner(
        _build_planner_msg(prompt, blueprint, workspace, template_key, app_name),
        backend="gemini", api_key=api_key, model=model,
    )
    _log_plan(plan, log_fn)
    total_tokens = 0

    if cancel_fn and cancel_fn():
        return "Stopped by user.", 0

    design_summary = ""
    if not _should_skip_design(plan):
        if log_fn:
            log_fn("info", "━━━ Design system phase ━━━")
            log_fn("thinking", "Preparing design system…")
        design_summary, design_tokens = _gemini_loop(
            api_key, model, _DESIGN_AGENT_SYSTEM, workspace,
            _design_agent_user_msg(app_name, template_key, plan, prompt, workspace),
            log_fn, cancel_fn, max_iterations=_ITERS_DESIGN,
        )
        total_tokens += design_tokens
        if cancel_fn and cancel_fn():
            return design_summary or "Stopped by user.", total_tokens
    elif log_fn:
        log_fn("info", "Design phase skipped — no new components or theme changes.")

    if log_fn:
        log_fn("info", "━━━ Execution phase ━━━")
    exec_summary, exec_tokens = _gemini_loop(
        api_key, model, _UPDATE_SYSTEM, workspace,
        _update_user_msg_with_brief(app_name, template_key, blueprint, prompt, plan, workspace, design_summary),
        log_fn, cancel_fn, max_iterations=_ITERS_EXEC,
    )
    total_tokens += exec_tokens
    if cancel_fn and cancel_fn():
        return exec_summary or "Stopped by user.", total_tokens

    if log_fn:
        log_fn("info", "━━━ Validation phase ━━━")
        log_fn("thinking", "Validating implementation…")
    val_summary, val_tokens = _gemini_loop(
        api_key, model, _VALIDATOR_SYSTEM, workspace,
        _validator_user_msg(app_name, template_key, plan, prompt, workspace, exec_summary),
        log_fn, cancel_fn, max_iterations=_ITERS_VALIDATE,
    )
    total_tokens += val_tokens
    if cancel_fn and cancel_fn():
        return val_summary or exec_summary, total_tokens

    if _validation_needs_fix_pass(val_summary):
        if log_fn:
            log_fn("info", "━━━ Post-validation fix pass ━━━")
            log_fn("thinking", "Validator found unresolved issues — running targeted fix pass…")
        fix_summary, fix_tokens = _gemini_loop(
            api_key, model, _UPDATE_SYSTEM, workspace,
            _make_fix_prompt(val_summary), log_fn, cancel_fn, max_iterations=_ITERS_FIX,
        )
        total_tokens += fix_tokens
        val_summary = fix_summary
        if cancel_fn and cancel_fn():
            return val_summary or exec_summary, total_tokens

    if not _should_skip_security(plan):
        if log_fn:
            log_fn("info", "━━━ Security review phase ━━━")
            log_fn("thinking", "Reviewing for security issues…")
        sec_summary, sec_tokens = _gemini_loop(
            api_key, model, _SECURITY_SYSTEM, workspace,
            _security_user_msg(app_name, template_key, plan, prompt, workspace),
            log_fn, cancel_fn, max_iterations=_ITERS_SECURITY,
        )
        total_tokens += sec_tokens
    elif log_fn:
        log_fn("info", "Security phase skipped — no auth/API/storage changes.")

    return val_summary or exec_summary, total_tokens


# ---------------------------------------------------------------------------
# OpenAI / GPT agent loop
# ---------------------------------------------------------------------------

def _openai_loop(
    api_key: str,
    model: str,
    system: str,
    workspace: Path,
    initial_user_msg: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
    max_iterations: int = _MAX_ITERATIONS,
) -> tuple[str, int]:
    """OpenAI tool-use agent loop. Returns (summary, total_tokens)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    openai_tools = _tools_to_ollama_format(TOOLS)  # same OpenAI-compatible format
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": initial_user_msg},
    ]
    last_text = ""
    write_calls = 0
    pushback_sent = False
    total_tokens = 0

    for iteration in range(max_iterations):
        if cancel_fn and cancel_fn():
            if log_fn:
                log_fn("warning", "Agent stopped by user.")
            return "Stopped by user.", 0
        warn_at = max(1, max_iterations - 10)
        if iteration == warn_at and log_fn:
            log_fn("warning", f"Build is complex ({iteration} steps so far) — finishing up…")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                max_tokens=8096,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI agent request failed: {exc}") from exc

        if response.usage:
            total_tokens += response.usage.total_tokens

        msg = response.choices[0].message
        # Append as dict so it's serialisable for the next round
        messages.append({"role": "assistant", "content": msg.content or "", **({"tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (msg.tool_calls or [])
        ]} if msg.tool_calls else {})})

        tool_calls = msg.tool_calls or []
        text = msg.content or ""
        done_text: str | None = None

        if text:
            last_text = text
            if log_fn and text.strip():
                log_fn("text", text.strip()[:160])
            if "DONE" in text.upper():
                done_text = text

        if done_text is not None and not tool_calls:
            if write_calls == 0 and not pushback_sent:
                pushback_sent = True
                if log_fn:
                    log_fn("info", "Agent said DONE without writing files — asking it to implement…")
                messages.append({"role": "user", "content": (
                    "You said you were done but haven't called write_file yet. "
                    "Implement the changes now using write_file."
                )})
                continue
            return done_text, total_tokens

        finish_reason = response.choices[0].finish_reason
        if finish_reason == "stop" and not tool_calls:
            if write_calls == 0 and not pushback_sent:
                pushback_sent = True
                if log_fn:
                    log_fn("info", "No files written yet — asking agent to write the code…")
                messages.append({"role": "user", "content": (
                    "You haven't written any files yet. Use write_file to implement the changes."
                )})
                continue
            return last_text or "Done.", total_tokens

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}
            if tool_name == "write_file":
                write_calls += 1
                if log_fn:
                    log_fn("file_written", tool_input.get("path", ""))
            if log_fn:
                log_fn("tool", tool_message(tool_name, tool_input))
            result = execute_tool(tool_name, tool_input, workspace, log_fn)
            if len(result) > _TOOL_RESULT_LIMIT:
                result = result[:_TOOL_RESULT_LIMIT] + "\n…[truncated]"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    if log_fn:
        log_fn("warning", "Agent reached iteration limit.")
    return "Agent reached iteration limit.", total_tokens


def run_build_agent_openai(
    workspace: Path,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> tuple[str, int]:
    system = _build_system(template_key)
    user_msg = (
        f"App name: {app_name}\nTemplate: {template_key}\n\n"
        f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
        "Build this application now following the phases in your instructions. "
        "Narrate each step. When finished, write DONE: <summary of what was built>."
    )
    return _openai_loop(api_key, model, system, workspace, user_msg, log_fn)


def run_update_agent_openai(
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Plan → design? → execute → validate → security? using OpenAI/GPT."""
    if log_fn:
        log_fn("thinking", "Planning changes…")
    plan = _call_planner(
        _build_planner_msg(prompt, blueprint, workspace, template_key, app_name),
        backend="gpt", api_key=api_key, model=model,
    )
    _log_plan(plan, log_fn)
    total_tokens = 0

    if cancel_fn and cancel_fn():
        return "Stopped by user.", 0

    design_summary = ""
    if not _should_skip_design(plan):
        if log_fn:
            log_fn("info", "━━━ Design system phase ━━━")
            log_fn("thinking", "Preparing design system…")
        design_summary, design_tokens = _openai_loop(
            api_key, model, _DESIGN_AGENT_SYSTEM, workspace,
            _design_agent_user_msg(app_name, template_key, plan, prompt, workspace),
            log_fn, cancel_fn, max_iterations=_ITERS_DESIGN,
        )
        total_tokens += design_tokens
        if cancel_fn and cancel_fn():
            return design_summary or "Stopped by user.", total_tokens
    elif log_fn:
        log_fn("info", "Design phase skipped — no new components or theme changes.")

    if log_fn:
        log_fn("info", "━━━ Execution phase ━━━")
    exec_summary, exec_tokens = _openai_loop(
        api_key, model, _UPDATE_SYSTEM, workspace,
        _update_user_msg_with_brief(app_name, template_key, blueprint, prompt, plan, workspace, design_summary),
        log_fn, cancel_fn, max_iterations=_ITERS_EXEC,
    )
    total_tokens += exec_tokens
    if cancel_fn and cancel_fn():
        return exec_summary or "Stopped by user.", total_tokens

    if log_fn:
        log_fn("info", "━━━ Validation phase ━━━")
        log_fn("thinking", "Validating implementation…")
    val_summary, val_tokens = _openai_loop(
        api_key, model, _VALIDATOR_SYSTEM, workspace,
        _validator_user_msg(app_name, template_key, plan, prompt, workspace, exec_summary),
        log_fn, cancel_fn, max_iterations=_ITERS_VALIDATE,
    )
    total_tokens += val_tokens
    if cancel_fn and cancel_fn():
        return val_summary or exec_summary, total_tokens

    if _validation_needs_fix_pass(val_summary):
        if log_fn:
            log_fn("info", "━━━ Post-validation fix pass ━━━")
            log_fn("thinking", "Validator found unresolved issues — running targeted fix pass…")
        fix_summary, fix_tokens = _openai_loop(
            api_key, model, _UPDATE_SYSTEM, workspace,
            _make_fix_prompt(val_summary), log_fn, cancel_fn, max_iterations=_ITERS_FIX,
        )
        total_tokens += fix_tokens
        val_summary = fix_summary
        if cancel_fn and cancel_fn():
            return val_summary or exec_summary, total_tokens

    if not _should_skip_security(plan):
        if log_fn:
            log_fn("info", "━━━ Security review phase ━━━")
            log_fn("thinking", "Reviewing for security issues…")
        sec_summary, sec_tokens = _openai_loop(
            api_key, model, _SECURITY_SYSTEM, workspace,
            _security_user_msg(app_name, template_key, plan, prompt, workspace),
            log_fn, cancel_fn, max_iterations=_ITERS_SECURITY,
        )
        total_tokens += sec_tokens
    elif log_fn:
        log_fn("info", "Security phase skipped — no auth/API/storage changes.")

    return val_summary or exec_summary, total_tokens


# ---------------------------------------------------------------------------
# Fix agents — executor-only, no pipeline
# Used by build_worker auto-fix loop to avoid running 5 agents per compile error.
# ---------------------------------------------------------------------------

def run_fix_agent(
    workspace: Path,
    prompt: str,
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Executor-only fix pass using Claude/Anthropic — no planner, design, validator, or security."""
    import anthropic as _anthropic
    client = _anthropic.Anthropic(api_key=api_key, timeout=180.0)
    return _loop(client, model, _UPDATE_SYSTEM, workspace, prompt, log_fn, cancel_fn, max_iterations=_ITERS_FIX)


def run_fix_agent_ollama(
    workspace: Path,
    prompt: str,
    app_name: str,
    template_key: str,
    base_url: str,
    model: str,
    timeout: int = 300,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Executor-only fix pass using Ollama/Qwen3."""
    return _ollama_loop(base_url, model, _UPDATE_SYSTEM, workspace, prompt, timeout, log_fn, cancel_fn, max_iterations=_ITERS_FIX)


def run_fix_agent_gemini(
    workspace: Path,
    prompt: str,
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Executor-only fix pass using Gemini."""
    return _gemini_loop(api_key, model, _UPDATE_SYSTEM, workspace, prompt, log_fn, cancel_fn, max_iterations=_ITERS_FIX)


def run_fix_agent_openai(
    workspace: Path,
    prompt: str,
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    """Executor-only fix pass using OpenAI/GPT."""
    return _openai_loop(api_key, model, _UPDATE_SYSTEM, workspace, prompt, log_fn, cancel_fn, max_iterations=_ITERS_FIX)


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
    cancel_fn: Callable[[], bool] | None = None,
    max_iterations: int = _MAX_ITERATIONS,
) -> tuple[str, int]:
    """Agent tool loop. Returns (summary, total_tokens_used)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_user_msg}]
    total_tokens = 0
    write_calls = 0
    pushback_sent = False

    for iteration in range(max_iterations):
        if cancel_fn and cancel_fn():
            if log_fn:
                log_fn("warning", "Agent stopped by user.")
            return "Stopped by user.", 0
        warn_at = max(1, max_iterations - 10)
        if iteration == warn_at and log_fn:
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
                    if log_fn:
                        log_fn("file_written", (block.input or {}).get("path", ""))
                msg = tool_message(block.name, block.input)
                logger.debug("Tool %s → %s", block.name, msg)
                if log_fn:
                    log_fn("tool", msg)
                result = execute_tool(block.name, block.input, workspace, log_fn)
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
        log_fn("warning", "Agent reached iteration limit.")
    return "Agent reached iteration limit.", total_tokens
