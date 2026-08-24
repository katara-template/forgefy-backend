"""Sync Redis publisher for build/update log events.

Workers call make_log_publisher() to get a callable that streams
events to the channel  build:{project_id}:logs  via Redis pub/sub.
The WebSocket endpoint /ws/projects/{project_id}/logs subscribes to
this channel and forwards the events to the browser.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Descriptive write_file labels
# ---------------------------------------------------------------------------

def _feature_from_path(p: str) -> str:
    """Extract the feature name from a features/{feature}/... path."""
    parts = p.replace("\\", "/").split("/")
    try:
        idx = parts.index("features")
        raw = parts[idx + 1] if idx + 1 < len(parts) else ""
        return raw.replace("_", " ").replace("-", " ").title()
    except ValueError:
        return ""


def _describe_write(path: str) -> str:
    """Produce a human-readable label for a write_file call from the file path."""
    p = path.lower().replace("\\", "/")
    stem = Path(path).stem
    name = stem.replace("_", " ").replace("-", " ").title()
    filename = Path(path).name.lower()
    feature = _feature_from_path(p)
    feat_label = f" [{feature}]" if feature else ""

    # ── Clean Architecture — Flutter domain layer ─────────────────────────────
    if "/domain/entities/" in p:
        return f"Domain entity{feat_label} · {name}"
    if "/domain/repositories/" in p:
        return f"Domain repository{feat_label} · {name}"
    if "/domain/usecases/" in p:
        return f"Use-case{feat_label} · {name}"

    # ── Clean Architecture — Flutter data layer ───────────────────────────────
    if "/data/models/" in p:
        return f"Data model{feat_label} · {name}"
    if "/data/datasources/" in p:
        kind = "remote" if "remote" in p else "local"
        return f"Datasource ({kind}){feat_label} · {name}"
    if "/data/repositories/" in p:
        return f"Repository impl{feat_label} · {name}"

    # ── BLoC ─────────────────────────────────────────────────────────────────
    if "/presentation/bloc/" in p or "/bloc/" in p:
        if "_event" in p:
            return f"BLoC events{feat_label} · {name}"
        if "_state" in p:
            return f"BLoC states{feat_label} · {name}"
        return f"BLoC{feat_label} · {name}"

    # ── Redux / slice (React Native feature-sliced) ───────────────────────────
    if "/slice/" in p:
        return f"Redux slice{feat_label} · {name}"
    if "/app/store" in p:
        return "Redux store · store.ts"
    if "/app/rootreducer" in p or "rootreducer" in p:
        return "Redux root reducer"
    if p.endswith("_slice.ts") or p.endswith("slice.ts"):
        return f"Redux slice · {name}"

    # ── Next.js API Route Handlers (app/api/**/route.ts) ─────────────────────
    if "/api/" in p and ("route.ts" in p or "route.js" in p):
        # Derive resource name from the path segments between /api/ and /route
        parts = p.replace("\\", "/").split("/")
        try:
            api_idx = parts.index("api")
            resource_parts = [
                s for s in parts[api_idx + 1:]
                if s not in ("route.ts", "route.js", "[id]", "[slug]", "")
            ]
            resource = " / ".join(p.replace("_", " ").replace("-", " ").title()
                                  for p in resource_parts)
        except (ValueError, IndexError):
            resource = name
        is_dynamic = "[id]" in p or "[slug]" in p
        suffix = " [id]" if is_dynamic else ""
        return f"API route{suffix} · {resource or name}"

    # ── RTK Query / feature API ───────────────────────────────────────────────
    if "/features/" in p and "/api/" in p:
        return f"API endpoints{feat_label} · {name}"

    # ── Feature hooks ─────────────────────────────────────────────────────────
    if "/features/" in p and "/hooks/" in p:
        return f"Hook{feat_label} · {name}"

    # ── Feature types ─────────────────────────────────────────────────────────
    if "/features/" in p and "/types/" in p:
        return f"Types{feat_label} · {name}"

    # ── Screens / pages ───────────────────────────────────────────────────────
    if any(seg in p for seg in ("/screens/", "/pages/", "/presentation/pages/")):
        return f"Screen{feat_label} · {name}"
    if any(p.endswith(s) for s in ("_screen.dart", "_screen.tsx", "_screen.jsx",
                                    "_page.dart", "_page.tsx", "_page.jsx")):
        return f"Screen · {name}"
    if "/page.tsx" in p or "/page.jsx" in p:
        folder = Path(path).parent.name.replace("-", " ").replace("_", " ").title()
        return f"Page · {folder or name}"

    # ── Widgets / components ──────────────────────────────────────────────────
    if "/presentation/widgets/" in p:
        return f"Widget{feat_label} · {name}"
    if "/features/" in p and "/components/" in p:
        return f"Component{feat_label} · {name}"
    if any(seg in p for seg in ("/widgets/", "/components/", "/ui/",
                                  "widgets/", "components/", "ui/")):
        return f"Component · {name}"

    # ── Services ─────────────────────────────────────────────────────────────
    if "/services/" in p:
        return f"Service · {name}"
    if any(p.endswith(s) for s in ("_service.dart", "service.dart",
                                    "_service.ts", "service.ts")):
        return f"Service · {name}"

    # ── Core infrastructure ───────────────────────────────────────────────────
    if "/core/error/" in p:
        return f"Core · {name}"
    if "/core/network/" in p:
        return f"Network client · {name}"
    if "/core/usecases/" in p:
        return f"Core use-case base · {name}"
    if "/core/theme/" in p or "/core/utils/" in p:
        return f"Core · {name}"

    # ── Navigation ────────────────────────────────────────────────────────────
    if "/navigation/" in p:
        return f"Navigation · {name}"
    if any(p.endswith(s) for s in ("navigator.dart", "navigator.tsx",
                                    "_navigator.dart", "_navigator.tsx", "router.tsx")):
        return f"Navigation · {name}"

    # ── Shared hooks ─────────────────────────────────────────────────────────
    if "/hooks/" in p or p.startswith("hooks/"):
        return f"Hook · {name}"

    # ── Injection / DI ───────────────────────────────────────────────────────
    if "injection_container" in p or "di_container" in p or "locator" in p:
        return "Dependency injection · injection_container.dart"

    # ── State / providers ────────────────────────────────────────────────────
    if any(seg in p for seg in ("/providers/", "/provider/", "/store/", "/context/")):
        return f"State · {name}"

    # ── Types / interfaces ────────────────────────────────────────────────────
    if "/types/" in p or p.startswith("types/") or p.endswith("types.ts") or p.endswith("types.d.ts"):
        return f"Types · {name}"

    # ── Styles / theme ────────────────────────────────────────────────────────
    if any(seg in p for seg in ("/styles/", "/theme/", "/constants/")):
        return f"Styles · {name}"
    if any(kw in stem for kw in ("color", "theme", "style", "constant")):
        return f"Theme · {name}"

    # ── Utilities ─────────────────────────────────────────────────────────────
    if "/utils/" in p or "/helpers/" in p or "/lib/" in p:
        return f"Utility · {name}"

    # ── Well-known config files ────────────────────────────────────────────────
    # ── Next.js middleware ────────────────────────────────────────────────────
    if filename == "middleware.ts" or filename == "middleware.js":
        return "Auth middleware · middleware.ts"

    # ── Next.js server-only lib files ─────────────────────────────────────────
    if filename in ("db.ts", "db.js"):
        return "Database client · lib/db.ts"
    if filename in ("validations.ts", "validations.js", "schemas.ts"):
        return "Zod validation schemas · lib/validations.ts"

    _named: dict[str, str] = {
        "pubspec.yaml":       "Flutter dependencies · pubspec.yaml",
        "package.json":       "npm dependencies · package.json",
        "app.json":           "Expo config · app.json",
        "app.config.js":      "Expo config",
        "app.config.ts":      "Expo config",
        "tailwind.config.ts": "Tailwind config",
        "tailwind.config.js": "Tailwind config",
        "tsconfig.json":      "TypeScript config",
        "next.config.ts":     "Next.js config",
        "next.config.js":     "Next.js config",
        "main.dart":          "App entry · main.dart",
        "app.dart":           "App root widget · app.dart",
        "app.tsx":            "App root · App.tsx",
        "injection_container.dart": "Dependency injection · injection_container.dart",
        "layout.tsx":         "Layout · layout.tsx",
        "_layout.tsx":        "Root layout · _layout.tsx",
        "globals.css":        "Global styles",
        "roottreducer.ts":    "Redux root reducer",
        "rootreducer.ts":     "Redux root reducer",
        "store.ts":           "Redux store",
        "httpclient.ts":      "HTTP client · httpClient.ts",
        "api_client.dart":    "API client · api_client.dart",
        "auth.ts":            "Auth helpers · lib/auth.ts",
        "api.ts":             "API client · lib/api.ts",
        "utils.ts":           "Utilities · lib/utils.ts",
    }
    if filename in _named:
        return _named[filename]

    return f"Writing `{path}`"


# ---------------------------------------------------------------------------
# Per-user event publisher (quota_exceeded, etc.)
# ---------------------------------------------------------------------------

def publish_user_event(redis_url: str, user_id: str, event_type: str, message: str, **extra) -> None:
    """Publish a one-off event to the per-user Redis channel.

    The /ws/user/events WebSocket endpoint subscribes to this channel and
    forwards events to the browser in real-time.
    """
    try:
        import redis as _redis
        payload = json.dumps({"type": event_type, "message": message, **extra})
        r = _redis.from_url(redis_url, decode_responses=True)
        r.publish(f"user:{user_id}:events", payload)
    except Exception as exc:
        logger.debug("publish_user_event error: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point used by build_agent.py
# ---------------------------------------------------------------------------

def tool_message(tool_name: str, inputs: dict) -> str:
    """Return a human-readable label for a tool call."""
    match tool_name:
        case "write_file":
            return _describe_write(inputs.get("path", ""))

        case "edit_file":
            path = inputs.get("path", "")
            scope = "all occurrences" if inputs.get("replace_all") else "1 change"
            return f"Editing `{path}` · {scope}"

        case "read_file":
            path = inputs.get("path", "")
            if offset := inputs.get("offset"):
                return f"Reading `{path}` from line {offset}"
            return f"Reading `{path}`"

        case "grep":
            where = inputs.get("glob") or inputs.get("path") or "the project"
            return f"Searching for `{inputs.get('pattern', '')}` in {where}"

        case "glob":
            return f"Finding files matching `{inputs.get('pattern', '')}`"

        case "list_files":
            p = inputs.get("path", ".")
            suffix = "/" if not p.endswith("/") else ""
            return f"Exploring `{p}{suffix}`"

        case "create_directory":
            return f"Creating folder `{inputs.get('path', '')}/`"

        case "delete_file":
            return f"Removing `{inputs.get('path', '')}`"

        case "move_file":
            return f"Moving `{inputs.get('source', '')}` → `{inputs.get('destination', '')}`"

        case "run_command":
            cmd = " ".join(str(a) for a in (inputs.get("command") or []))
            note = inputs.get("description") or ""
            return f"Running `{cmd}`" + (f" · {note}" if note else "")

        case "job_start":
            cmd = " ".join(str(a) for a in (inputs.get("command") or []))
            note = inputs.get("description") or ""
            return f"Starting `{cmd}` in the background" + (f" · {note}" if note else "")

        case "job_output":
            return f"Checking background job {inputs.get('job_id', '')}"

        case "job_kill":
            return f"Stopping background job {inputs.get('job_id', '')}"

        case "todo_write":
            todos = inputs.get("todos") or []
            done = sum(1 for t in todos if isinstance(t, dict) and t.get("status") == "completed")
            current = next(
                (
                    t.get("active_form") or t.get("content")
                    for t in todos
                    if isinstance(t, dict) and t.get("status") == "in_progress"
                ),
                "",
            )
            label = f"Task list · {done}/{len(todos)} done"
            return f"{label} · {current}" if current else label

        case "generate_image":
            return f"Generating image · {inputs.get('filename', '')}"

        case "generate_video":
            return f"Generating video · {inputs.get('filename', '')}"

        case _:
            return tool_name


# ---------------------------------------------------------------------------
# Redis publisher
# ---------------------------------------------------------------------------

_LOG_HISTORY_MAX = 200   # entries kept in the Redis list per project
_LOG_HISTORY_TTL = 7200  # seconds — 2 hours


def make_log_publisher(project_id: str, redis_url: str):
    """Return a sync callable that publishes build log events.

    Each event is:
      1. Published on pub/sub channel  build:{project_id}:logs  (live subscribers)
      2. Appended to the Redis list    build:{project_id}:log_history  (replay buffer)

    Event JSON:  { "type": "<event_type>", "message": "<text>" }
    Event types: started | info | thinking | tool | text | warning | error | done
                 | file_written | todo

    `todo` carries a JSON array of {content, status, active_form} objects — the
    agent's current task list, replacing whatever it sent before, so the client
    can render a live checklist rather than a spinner.
    """
    channel = f"build:{project_id}:logs"
    history_key = f"build:{project_id}:log_history"
    _client: object = None

    def _get_client():
        nonlocal _client
        if _client is None:
            import redis
            _client = redis.from_url(redis_url, decode_responses=True)
        return _client

    def publish(event_type: str, message: str) -> None:
        try:
            payload = json.dumps({"type": event_type, "message": message})
            r = _get_client()
            pipe = r.pipeline()
            pipe.publish(channel, payload)
            pipe.rpush(history_key, payload)
            pipe.ltrim(history_key, -_LOG_HISTORY_MAX, -1)
            pipe.expire(history_key, _LOG_HISTORY_TTL)
            pipe.execute()
        except Exception as exc:
            logger.debug("build_logger publish error: %s", exc)

    return publish
