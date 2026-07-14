"""Projects endpoints — list, get, and dispatch prompt-driven updates."""
from __future__ import annotations

import json
import logging
import secrets
import uuid
from contextlib import suppress
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter

from app.core.dispatch import dispatch
from app.core.exceptions import ForbiddenError, NotFoundError, ValidationError
from app.deps import CurrentUser, DBSession, SettingsDep
from app.schemas.project import (
    ChatHistoryRequest,
    ChatRequest,
    ChatResponse,
    ConnectSupabaseRequest,
    ProjectOut,
    UpdateProjectRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _doc_to_out(doc) -> ProjectOut:
    d = doc.to_dict()
    return ProjectOut(
        id=uuid.UUID(doc.id),
        owner_id=uuid.UUID(d["owner_id"]),
        app_name=d["app_name"],
        template_key=d["template_key"],
        repo_full_name=d["repo_full_name"],
        github_url=d["github_url"],
        repo_owner=d.get("repo_owner"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        session_id=uuid.UUID(d["session_id"]) if d.get("session_id") else None,
        blueprint_id=uuid.UUID(d["blueprint_id"]) if d.get("blueprint_id") else None,
        preview_url=d.get("preview_url"),
        artifact_url=d.get("artifact_url"),
        is_updating=d.get("is_updating", False),
        build_error=d.get("build_error"),
        build_error_action=d.get("build_error_action"),
        supabase_project_ref=d.get("supabase_project_ref"),
        supabase_url=d.get("supabase_url"),
        supabase_anon_key=d.get("supabase_anon_key"),
        neon_project_id=d.get("neon_project_id"),
        neon_data_api_url=d.get("neon_data_api_url"),
        firebase_project_id=d.get("firebase_project_id"),
        firebase_api_key=d.get("firebase_api_key"),
        firebase_auth_domain=d.get("firebase_auth_domain"),
        firebase_storage_bucket=d.get("firebase_storage_bucket"),
        firebase_messaging_sender_id=d.get("firebase_messaging_sender_id"),
        firebase_app_id=d.get("firebase_app_id"),
        db_decision_pending=d.get("db_decision_pending", False),
        db_decision_reason=d.get("db_decision_reason"),
    )


async def _get_owned(project_id: uuid.UUID, user_id: uuid.UUID, db) -> ProjectOut:
    doc = await db.collection("projects").document(str(project_id)).get()
    if not doc.exists:
        raise NotFoundError(f"Project {project_id} not found")
    d = doc.to_dict()
    if uuid.UUID(d["owner_id"]) != user_id:
        raise ForbiddenError("Access denied")
    return _doc_to_out(doc)


async def _finalize_db_connect(project_id: uuid.UUID, project: ProjectOut, db) -> dict:
    """Called by every connect_* endpoint right after a database provisions.

    Two cases, per the "never touch a database without consent" rule:
    - The initial build was withheld pending this exact decision (db_decision_pending) —
      clear the flag and finally dispatch the build the user already approved.
    - Otherwise this is an already-built project — connecting only provisions the
      database; a separate explicit confirmation (POST /{id}/wire-database) is required
      before any code gets rewritten to use it.
    """
    if project.db_decision_pending:
        await db.collection("projects").document(str(project_id)).update({
            "db_decision_pending": False,
            "db_decision_reason": None,
            "is_updating": True,
            "updated_at": datetime.now(UTC),
        })
        from app.workers.build_worker import run_build
        await dispatch(run_build, args=[str(project.session_id), str(project_id)], queue="build")
        return {"build_queued": True}

    return {"prompt_wire_in": True}


@router.get("", response_model=list[ProjectOut])
async def list_projects(db: DBSession, user: CurrentUser) -> list[ProjectOut]:
    """Return all projects owned by the current user."""
    docs = (
        await db.collection("projects")
        .where("owner_id", "==", str(user.id))
        .get()
    )
    projects = sorted(
        [_doc_to_out(d) for d in docs],
        key=lambda p: p.updated_at,
        reverse=True,
    )
    return projects


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> ProjectOut:
    """Return a single project."""
    return await _get_owned(project_id, user.id, db)


@router.post("/{project_id}/stop", response_model=dict)
async def stop_project(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Signal any running agent or build task for this project to stop."""
    await _get_owned(project_id, user.id, db)  # ownership check

    from app.build.cancel import request_cancel
    from app.config import get_settings
    settings = get_settings()
    # request_cancel opens a sync Redis connection — keep it off the event loop.
    await anyio.to_thread.run_sync(request_cancel, settings.REDIS_URL, str(project_id))

    # Immediately mark as not-updating so the UI unblocks
    from datetime import datetime
    await db.collection("projects").document(str(project_id)).update({
        "is_updating": False,
        "updated_at": datetime.now(UTC),
    })
    return {"stopped": True}


@router.delete("/{project_id}", response_model=dict)
async def delete_project(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Permanently delete a project and its associated resources."""

    project = await _get_owned(project_id, user.id, db)

    if project.is_updating:
        raise ValidationError("A build is in progress — stop it before deleting.")

    pid = str(project_id)

    # 1. Delete Firestore documents
    await db.collection("projects").document(pid).delete()
    with suppress(Exception):  # non-fatal
        await db.collection("project_chats").document(pid).delete()

    # 2. Try to delete the GitHub repo (needs delete_repo scope — may fail gracefully)
    if project.repo_full_name:
        try:
            import httpx

            from app.build.github_token import get_valid_github_token
            from app.config import get_settings
            settings = get_settings()
            token = await get_valid_github_token(str(user.id), settings.GITHUB_TOKEN)
            async with httpx.AsyncClient(timeout=10) as client:
                await client.delete(
                    f"https://api.github.com/repos/{project.repo_full_name}",
                    headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
                )
        except Exception as exc:
            logger.warning("Could not delete GitHub repo %s: %s", project.repo_full_name, exc)

    # 3. Clear any Redis cancel flag left over
    try:
        from app.build.cancel import clear_cancel
        from app.config import get_settings
        settings = get_settings()
        clear_cancel(settings.REDIS_URL, pid)
    except Exception:
        pass

    logger.info("Project deleted project=%s user=%s repo=%s", pid, user.id, project.repo_full_name)
    return {"deleted": True}


_CODE_SKIP = frozenset({
    "node_modules", ".dart_tool", "build", ".gradle", ".idea",
    "__pycache__", ".next", "dist", ".expo", "android", "ios",
    ".git", "coverage",
})
_CODE_SKIP_EXT = frozenset({
    ".lock", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".mp4", ".webp", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pyc", ".class", ".jar", ".aar", ".apk", ".so",
})


@router.get("/{project_id}/code/tree")
async def get_code_tree(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Return a flat list of source file paths from the project's GitHub repo."""
    import httpx

    from app.build.github_token import get_valid_github_token
    from app.config import get_settings

    project = await _get_owned(project_id, user.id, db)
    if not project.repo_full_name:
        return {"files": []}

    settings = get_settings()
    token = await get_valid_github_token(str(user.id), settings.GITHUB_TOKEN)
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    async with httpx.AsyncClient(timeout=15) as client:
        # Try main branch first, then master
        tree_data = None
        for branch in ("main", "master"):
            r = await client.get(
                f"https://api.github.com/repos/{project.repo_full_name}/git/trees/{branch}?recursive=1",
                headers=headers,
            )
            if r.status_code == 200:
                tree_data = r.json()
                break

    if not tree_data:
        return {"files": []}

    files = []
    for item in tree_data.get("tree", []):
        if item.get("type") != "blob":
            continue
        path: str = item["path"]
        parts = path.split("/")
        # Skip build/vendor dirs and binary extensions
        if any(p in _CODE_SKIP for p in parts):
            continue
        ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
        if ext in _CODE_SKIP_EXT:
            continue
        files.append(path)

    # .env.local is gitignored so it won't be in the GitHub tree.
    # Inject it as a virtual file if the build saved its placeholder content.
    if getattr(project, "env_local_template", None):
        files.append(".env.local")

    return {"files": sorted(files)}


@router.get("/{project_id}/code/file")
async def get_code_file(
    project_id: uuid.UUID,
    path: str,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Return the text content of a single file from the project's GitHub repo."""
    import base64

    import httpx

    from app.build.github_token import get_valid_github_token
    from app.config import get_settings

    project = await _get_owned(project_id, user.id, db)
    if not project.repo_full_name:
        raise NotFoundError("No repository linked to this project")

    # .env.local is gitignored — serve from the snapshot saved at build time
    if path == ".env.local":
        content = getattr(project, "env_local_template", None)
        if not content:
            raise NotFoundError(".env.local was not captured for this project")
        return {"path": path, "content": content}

    settings = get_settings()
    token = await get_valid_github_token(str(user.id), settings.GITHUB_TOKEN)
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"https://api.github.com/repos/{project.repo_full_name}/contents/{path}",
            headers=headers,
        )

    if r.status_code == 404:
        raise NotFoundError(f"File not found: {path}")
    if not r.is_success:
        raise NotFoundError(f"Could not fetch file: {r.status_code}")

    data = r.json()
    content_b64 = data.get("content", "")
    try:
        content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        content = "(binary file)"

    return {"path": path, "content": content}


@router.post("/{project_id}/update", response_model=dict)
async def update_project(
    project_id: uuid.UUID,
    body: UpdateProjectRequest,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Dispatch a prompt-driven update for the project. Returns immediately."""
    from app.core.usage import check_not_over_limit
    await check_not_over_limit(db, str(user.id))

    project = await _get_owned(project_id, user.id, db)

    if project.is_updating:
        raise ValidationError("A build or update is already in progress.")

    from app.workers.update_worker import apply_update
    await dispatch(
        apply_update,
        args=[str(project_id), body.prompt, str(user.id)],
        queue="build",
    )
    return {"status": "queued"}


@router.post("/{project_id}/chat", response_model=ChatResponse)
async def chat_with_project(
    project_id: uuid.UUID,
    body: ChatRequest,
    db: DBSession,
    user: CurrentUser,
) -> ChatResponse:
    """
    Deeply process the user's message before deciding what to do:
    - Greetings / questions → respond directly (no build queued)
    - Clear app-change requests → translate into a precise technical instruction, then queue
    - Vague / ambiguous requests → ask a clarifying question instead of wasting a build
    """
    from app.core.usage import check_not_over_limit
    await check_not_over_limit(db, str(user.id))

    project = await _get_owned(project_id, user.id, db)
    message = body.message.strip()

    if not message:
        return ChatResponse(type="chat", response="I didn't catch that — what would you like to do?")

    db_connected = bool(
        project.supabase_project_ref or project.neon_project_id or project.firebase_project_id
    )

    from app.config import get_settings
    from app.core.build_model import get_effective_build_model

    settings = get_settings()
    effective_model = await get_effective_build_model(db, settings, user_id=str(user.id))
    settings = settings.model_copy(update={"BUILD_MODEL": effective_model})
    fw = {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}.get(
        project.template_key, project.template_key
    )

    # Set up log publisher so the frontend sees progress in the build log panel
    from app.build.build_logger import make_log_publisher
    log_fn = make_log_publisher(str(project_id), settings.REDIS_URL)

    # ── Fast-path: long or clearly structured messages skip the classifier ──
    # Sending a 1000-word spec through the AI classifier risks JSON truncation
    # and model confusion. The planner in the build agent handles decomposition.
    import re as _re
    _is_structured = (
        len(message) > 600
        or bool(_re.search(r"^#{1,3} ", message, _re.MULTILINE))  # markdown headers
        or message.count("\n") > 8                                  # multi-line spec
    )
    if _is_structured:
        if project.is_updating:
            return ChatResponse(
                type="chat",
                response="An update is already in progress — I'll get to this as soon as it finishes.",
            )
        log_fn("started", f"Preparing update for {project.app_name}…")
        from app.workers.update_worker import apply_update
        await dispatch(
            apply_update,
            args=[str(project_id), message, str(user.id)],
            queue="build",
        )
        logger.info("Fast-path update queued project=%s msg_len=%d", project_id, len(message))
        return ChatResponse(
            type="update",
            response="Got it — I'll plan and implement everything now. Watch the build log for progress.",
            update_queued=True,
        )

    # Pull a compact summary of the blueprint so the classifier knows what
    # screens / features already exist in the app.
    doc = await db.collection("projects").document(str(project_id)).get()
    raw_project = doc.to_dict() or {}
    bp = raw_project.get("blueprint_context") or {}
    blueprint_summary = {
        "features": bp.get("features", []),
        "entities": bp.get("entities", []),
        "description": bp.get("description") or bp.get("app_description", ""),
        "stack": bp.get("stack", []),
    }

    # Pull recent conversation history so the classifier understands context and references
    chat_doc = await db.collection("project_chats").document(str(project_id)).get()
    recent_history = ""
    if chat_doc.exists:
        all_msgs = (chat_doc.to_dict() or {}).get("messages", [])
        prior_msgs = all_msgs[-10:]  # last 5 turns (user + assistant pairs)
        if prior_msgs:
            lines = []
            for m in prior_msgs:
                role = m.get("role", "user")
                text = (m.get("text") or "")[:300].strip()
                if not text or role == "error":
                    continue
                lines.append(f'{"User" if role == "user" else "You"}: {text}')
            if lines:
                recent_history = "\n".join(lines)

    system = f"""You are the Forgefy AI assistant for "{project.app_name}", a {fw} app.

App context (what already exists):
{json.dumps(blueprint_summary, indent=2)}
{f"""
Recent conversation (use this to understand references and follow-on requests):
{recent_history}
""" if recent_history else ""}

──────────────────────────────────────────────
YOUR JOB
──────────────────────────────────────────────
Analyse the user's message and reply with ONE of three types.

IMPORTANT: The build system uses a dedicated planner + executor pipeline.
The planner will automatically break any complex request into atomic steps.
You do NOT need to ask the user to narrow down scope — just pass the full
request through as the update_prompt. Big requests are fine.

1. "chat"
   Use for: greetings, thanks, questions about the app, status checks.
   response: a short, friendly reply.

2. "clarify"
   Use ONLY when the request is so ambiguous that even a planner cannot
   infer what to build. Reserved for truly meaningless messages like:
   "make it better", "fix it", "do something cool".
   DO NOT use for broad-but-specific requests (multiple features, full flows,
   comprehensive improvements). Those are "update".
   response: ask ONE specific question.
   clarify_options: REQUIRED whenever you use "clarify" (see ASKING CLARIFYING
   QUESTIONS below) — the user picks an answer, they never type a free-form reply.

3. "update"
   Use for ANY request that names features, screens, behaviour, or improvements —
   no matter how large or how many items are listed. When in doubt, choose "update".
   response: a short, enthusiastic confirmation of what will be built.
   update_prompt: pass the user's request through with full detail, expanding
   any implicit requirements so the planner has everything it needs.

──────────────────────────────────────────────
DATABASE CONSENT — never assume, always ask
──────────────────────────────────────────────
This project currently {"HAS" if db_connected else "has NO"} database connected.
{"" if db_connected else '''
If fulfilling this request would require durably storing structured data across
sessions/devices/users (saved records, user accounts, orders, posts, chat history,
bookmarks, etc.) — and NOT just simple local/in-memory/on-device state — do NOT
silently plan to add one. Instead:
  - type: "clarify"
  - response: a short note naming what needs storing and asking if they want
    to connect a database first, e.g. "This needs somewhere to save that data.
    Want to connect a database first?"
  - needs_database: true
  - do NOT set update_prompt and do NOT queue anything in this case.

If the user's message is clearly DECLINING a database you previously suggested
(a short reply like "no", "not now", "skip it", "continue without one" — check
the recent conversation above to see if your last message asked this question),
do not ask again. Instead proceed with the ORIGINAL feature request (look back
at the message before the decline) as a normal "update", using local/in-memory/
on-device storage instead of a real database, and set needs_database: false.
'''}
Otherwise (a database is already connected, or the request doesn't need durable
storage), proceed normally and set needs_database: false.

──────────────────────────────────────────────
ASKING CLARIFYING QUESTIONS — never free text, always a real choice
──────────────────────────────────────────────
Only ask when you actually need to (see "clarify" above — genuinely ambiguous
requests only, not broad-but-specific ones). When you do ask, the user picks an
answer by tapping a button — they can never type a free-form reply — so every
clarifying question MUST be phrased as EITHER:
  - a yes/no question, with clarify_options: ["Yes", "No"] (or two short
    domain-specific labels that mean the same thing, e.g. ["Grid", "List"]), OR
  - a multiple-choice question with EXACTLY three concrete, mutually exclusive
    options, with clarify_options: ["<option 1>", "<option 2>", "<option 3>"]
Never leave clarify_options empty or omit it when type is "clarify" — an
open-ended question the user has to type an answer to is not allowed. Keep each
option short (2-5 words) so it reads well as a button label.
This does not apply to the DATABASE CONSENT case above, which always uses its
own fixed Yes/No via needs_database — do not also set clarify_options there.

──────────────────────────────────────────────
HOW TO WRITE A GOOD update_prompt
──────────────────────────────────────────────
Expand the user's words into a thorough technical description.
Include every feature they listed. The planner will sequence the work.

If the user says: "add event discovery, creation, and user profiles"
Write: "Add the following features to the {fw} app:
        1. Event Discovery — browsable list of events with search/filter by category,
           date, and location. Tapping an event opens a detail screen.
        2. Event Creation — form flow for organisers: title, description, date/time,
           location, ticket type (free/paid), cover image upload.
        3. User Profiles — profile screen showing name, avatar, upcoming events,
           and past events. Editable via a settings sheet."

Always expand abbreviated requests into full feature descriptions.
Always name {fw}-specific patterns (widgets, hooks, routes) where you know them.

──────────────────────────────────────────────
OUTPUT — reply ONLY with valid JSON, no extra text:
{{
  "type": "chat" | "clarify" | "update",
  "response": "<message to show the user>",
  "update_prompt": "<detailed technical instruction — only present when type is update>",
  "needs_database": true | false,
  "clarify_options": ["Yes", "No"] | ["<option 1>", "<option 2>", "<option 3>"] | null
}}"""

    log_fn("thinking", "Analysing your request…")

    try:
        if settings.BUILD_MODEL == "Qwen3":
            import asyncio

            import requests as _req

            def _ollama_classify():
                r = _req.post(
                    f"{settings.OLLAMA_URL.rstrip('/')}/api/chat",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": message},
                        ],
                        "stream": False,
                    },
                    timeout=120,
                )
                r.raise_for_status()
                return (r.json().get("message") or {}).get("content", "").strip()

            raw = await asyncio.to_thread(_ollama_classify)

        elif settings.BUILD_MODEL == "gemini":
            import asyncio

            import requests as _req

            def _gemini_classify():
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent"
                r = _req.post(
                    url,
                    params={"key": settings.GEMINI_API_KEY},
                    json={
                        "system_instruction": {"parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": message}]}],
                        "generationConfig": {"maxOutputTokens": 1024},
                    },
                    timeout=60,
                )
                r.raise_for_status()
                parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts).strip()

            raw = await asyncio.to_thread(_gemini_classify)

        elif settings.BUILD_MODEL in ("gpt", "openai"):
            import asyncio

            def _openai_classify():
                from openai import OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": message},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()

            raw = await asyncio.to_thread(_openai_classify)

        else:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            ai_resp = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": message}],
            )
            raw = ai_resp.content[0].text.strip() if ai_resp.content else ""
    except Exception as exc:
        logger.error("Chat classifier failed project=%s: %s", project_id, exc, exc_info=True)
        return ChatResponse(type="chat", response="I'm having trouble processing that right now. Please try again in a moment.")

    # Strip <think>...</think> blocks that Qwen3 prepends
    import re as _re
    raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()

    # Strip markdown fences if the model wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Extract the first {...} JSON object in case there is surrounding text
    json_match = _re.search(r"\{.*\}", raw, flags=_re.DOTALL)
    raw = json_match.group(0) if json_match else raw

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Classifier returned invalid JSON — queue the raw message as an update so
        # it always reaches the AI agent rather than silently dropping it.
        if project.is_updating:
            return ChatResponse(
                type="chat",
                response="An update is already in progress — I'll queue this as soon as it finishes.",
            )
        log_fn("started", f"Preparing update for {project.app_name}…")
        from app.workers.update_worker import apply_update
        await dispatch(
            apply_update,
            args=[str(project_id), message, str(user.id)],
            queue="build",
        )
        return ChatResponse(type="update", response="On it — watch the build log for progress.", update_queued=True)

    intent = parsed.get("type", "chat")
    user_response = parsed.get("response", "")
    update_prompt = parsed.get("update_prompt", "").strip()
    needs_database = bool(parsed.get("needs_database", False))

    # A clarifying question must always be a real choice (yes/no or exactly three
    # options) — never free text. If the classifier didn't comply, drop the
    # options rather than render a broken/empty button row.
    raw_clarify_options = parsed.get("clarify_options")
    clarify_options: list[str] | None = None
    if (
        intent == "clarify"
        and isinstance(raw_clarify_options, list)
        and len(raw_clarify_options) in (2, 3)
        and all(isinstance(o, str) and o.strip() for o in raw_clarify_options)
    ):
        clarify_options = [o.strip() for o in raw_clarify_options]

    # Defensive: never queue an update alongside a pending database question,
    # even if the classifier mis-set both fields — asking always wins.
    if needs_database:
        return ChatResponse(type="clarify", response=user_response, needs_database=True)

    if intent == "update":
        if not update_prompt:
            # Classifier returned update type but no prompt — treat as clarify
            return ChatResponse(type="chat", response=user_response or "Could you be more specific about what you'd like changed?")

        if project.is_updating:
            return ChatResponse(
                type="chat",
                response="An update is already in progress — I'll queue this as soon as it finishes.",
            )

        log_fn("started", f"Preparing update for {project.app_name}…")
        from app.workers.update_worker import apply_update
        await dispatch(
            apply_update,
            args=[str(project_id), update_prompt, str(user.id)],
            queue="build",
        )
        logger.info("Queued update project=%s prompt_len=%d", project_id, len(update_prompt))
        return ChatResponse(type="update", response=user_response, update_queued=True)

    # "chat" or "clarify" — just return the response, nothing queued
    return ChatResponse(type=intent, response=user_response, clarify_options=clarify_options)


@router.get("/{project_id}/chat-history")
async def get_chat_history(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Return the persisted chat history for a project."""
    await _get_owned(project_id, user.id, db)
    doc = await db.collection("project_chats").document(str(project_id)).get()
    messages = (doc.to_dict() or {}).get("messages", []) if doc.exists else []
    return {"messages": messages}


@router.post("/{project_id}/chat-history")
async def save_chat_history(
    project_id: uuid.UUID,
    body: ChatHistoryRequest,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Persist the chat history for a project (last 100 messages)."""
    await _get_owned(project_id, user.id, db)
    from datetime import datetime
    messages = body.messages[-100:]
    await db.collection("project_chats").document(str(project_id)).set(
        {"messages": messages, "updated_at": datetime.now(UTC)},
    )
    return {"saved": len(messages)}


@router.post("/{project_id}/build-preview", response_model=dict)
async def trigger_preview_build(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Manually trigger a preview build and deployment for the project."""
    project = await _get_owned(project_id, user.id, db)

    if project.is_updating:
        raise ValidationError("A build or update is already in progress.")

    if not project.github_url:
        raise ValidationError("Project has no GitHub repository yet — wait for the initial build to finish.")

    from app.workers.build_worker import build_preview
    await dispatch(
        build_preview,
        args=[str(project_id), str(user.id)],
        queue="build",
    )
    return {"status": "queued"}


@router.post("/{project_id}/supabase/connect", response_model=dict)
async def connect_supabase(
    project_id: uuid.UUID,
    body: ConnectSupabaseRequest,
    db: DBSession,
    settings: SettingsDep,
    user: CurrentUser,
) -> dict:
    """Provision a real Supabase project for this Forgefy project.

    Requires the user's Supabase account to already be linked (see
    /api/v1/auth/supabase/authorize). The generated app's .env is updated
    with the real URL/anon key on the next build (see
    Workspace.write_supabase_env in app/build/workspace.py) — never the
    db_pass or a service_role key, which stay backend-only.
    """
    from app.build.supabase_token import get_valid_supabase_token
    from app.core.crypto import encrypt
    from app.integrations import supabase_management

    project = await _get_owned(project_id, user.id, db)

    token = await get_valid_supabase_token(db, str(user.id), settings)
    if not token:
        raise ValidationError("No Supabase account linked. Connect your Supabase account first.")

    db_pass = secrets.token_urlsafe(24)
    created = await supabase_management.create_project(
        token,
        organization_id=body.organization_id,
        name=project.app_name,
        db_pass=db_pass,
    )
    project_ref = created["id"]

    keys = await supabase_management.get_api_keys(token, project_ref)
    anon_key = next((k["api_key"] for k in keys if k.get("name") == "anon"), None)

    await db.collection("projects").document(str(project_id)).update({
        "supabase_project_ref": project_ref,
        "supabase_url": f"https://{project_ref}.supabase.co",
        "supabase_anon_key": anon_key,
        "supabase_db_pass": encrypt(db_pass),
        "supabase_connected_at": datetime.now(UTC),
    })
    extra = await _finalize_db_connect(project_id, project, db)
    return {"status": "connected", "project_ref": project_ref, **extra}


@router.post("/{project_id}/neon/connect", response_model=dict)
async def connect_neon(
    project_id: uuid.UUID,
    db: DBSession,
    settings: SettingsDep,
    user: CurrentUser,
) -> dict:
    """Provision a real Neon Postgres project for this Forgefy project.

    Embedded model — no user OAuth, uses Forgefy's own NEON_API_KEY. The raw
    Postgres connection string is a real secret and is encrypted and kept
    backend-only; only the Data API URL (public-safe) is ever written into the
    generated app's .env (see Workspace.write_neon_env in app/build/workspace.py).
    """
    from app.core.crypto import encrypt
    from app.integrations import neon_management

    if not settings.NEON_API_KEY:
        raise ValidationError("Neon integration is not configured on this server.")

    project = await _get_owned(project_id, user.id, db)

    created = await neon_management.create_project(
        settings.NEON_API_KEY, name=project.app_name, org_id=settings.NEON_ORG_ID or None
    )
    neon_project_id = created["project"]["id"]
    branch_id = created["branch"]["id"]
    database_name = created["databases"][0]["name"]
    connection_uri = created["connection_uris"][0]["connection_uri"]

    data_api = await neon_management.enable_data_api(
        settings.NEON_API_KEY,
        project_id=neon_project_id,
        branch_id=branch_id,
        database_name=database_name,
    )
    data_api_url = data_api["url"]

    await db.collection("projects").document(str(project_id)).update({
        "neon_project_id": neon_project_id,
        "neon_connection_uri": encrypt(connection_uri),
        "neon_data_api_url": data_api_url,
        "neon_connected_at": datetime.now(UTC),
    })
    extra = await _finalize_db_connect(project_id, project, db)
    return {"status": "connected", "project_id": neon_project_id, **extra}


def _make_gcp_project_id(app_name: str) -> str:
    """Build a globally-unique, spec-compliant GCP project id.

    GCP project ids must be 6-30 chars, lowercase letters/digits/hyphens, and
    unique across all of GCP — so a random suffix is required regardless of
    how unique app_name itself is.
    """
    import re

    slug = re.sub(r"[^a-z0-9-]", "-", app_name.lower()).strip("-")[:20] or "forgefy-app"
    return f"fg-{slug}-{secrets.token_hex(3)}"


@router.post("/{project_id}/firebase/connect", response_model=dict)
async def connect_firebase(
    project_id: uuid.UUID,
    db: DBSession,
    settings: SettingsDep,
    user: CurrentUser,
) -> dict:
    """Provision a real Firebase/Firestore project for this Forgefy project.

    Requires the user's Google account to already be linked (see
    /api/v1/auth/firebase/authorize). Firestore only — no Firebase Auth /
    Identity Platform setup, since choosing sign-in providers is a per-app
    product decision. The generated app's .env is updated with the real
    client config on the next build (see Workspace.write_firebase_env in
    app/build/workspace.py). The client config is public-safe by design
    (security enforced by Firestore Security Rules) so nothing here needs
    encryption, unlike Supabase's db_pass or Neon's connection string.
    """
    from app.build.firebase_token import get_valid_firebase_token
    from app.integrations import firebase_management

    project = await _get_owned(project_id, user.id, db)

    token = await get_valid_firebase_token(db, str(user.id), settings)
    if not token:
        raise ValidationError("No Google account linked. Connect your Google account first.")

    gcp_project_id = _make_gcp_project_id(project.app_name)
    config = await firebase_management.provision_project(
        token, project_id=gcp_project_id, display_name=project.app_name
    )
    # GCP always assigns the projectId we requested, but prefer whatever the API
    # actually confirms over the locally-generated value if the two ever diverge.
    firebase_project_id = config.get("projectId", gcp_project_id)

    await db.collection("projects").document(str(project_id)).update({
        "firebase_project_id": firebase_project_id,
        "firebase_api_key": config.get("apiKey"),
        "firebase_auth_domain": config.get("authDomain"),
        "firebase_storage_bucket": config.get("storageBucket"),
        "firebase_messaging_sender_id": config.get("messagingSenderId"),
        "firebase_app_id": config.get("appId"),
        "firebase_connected_at": datetime.now(UTC),
    })
    extra = await _finalize_db_connect(project_id, project, db)
    return {"status": "connected", "project_id": firebase_project_id, **extra}


@router.post("/{project_id}/skip-database", response_model=dict)
async def skip_database(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Decline the pre-build database suggestion and dispatch the withheld build.

    Only valid while a decision is actually pending (see approve_blueprint in
    app/api/v1/blueprints.py, which withholds the initial build when the
    classifier thinks the app needs persistent storage).
    """
    project = await _get_owned(project_id, user.id, db)

    if not project.db_decision_pending:
        raise ValidationError("No database decision is pending for this project.")

    await db.collection("projects").document(str(project_id)).update({
        "db_decision_pending": False,
        "db_decision_reason": None,
        "is_updating": True,
        "updated_at": datetime.now(UTC),
    })

    from app.workers.build_worker import run_build
    await dispatch(run_build, args=[str(project.session_id), str(project_id)], queue="build")
    return {"status": "queued"}


@router.post("/{project_id}/wire-database", response_model=dict)
async def wire_database(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Explicit second consent step: scan the app and wire the already-connected
    database into it. Connecting a database only provisions it — this is the
    separate confirmation required before any code gets rewritten to use it.
    """
    project = await _get_owned(project_id, user.id, db)

    if project.is_updating:
        raise ValidationError("A build or update is already in progress.")

    if project.supabase_project_ref:
        provider = "Supabase"
    elif project.neon_project_id:
        provider = "Neon"
    elif project.firebase_project_id:
        provider = "Firebase"
    else:
        raise ValidationError("No database is connected for this project yet.")

    prompt = (
        f"A {provider} database is now connected to this project. Create or extend "
        "the project's services/datasource layer for all reads and writes, and replace "
        "any mock/local/in-memory data handling with real calls through it. Scan the "
        "whole app for places that need updating — every screen or page that currently "
        "uses placeholder data should use the real database instead."
    )

    from app.workers.update_worker import apply_update
    await dispatch(apply_update, args=[str(project_id), prompt, str(user.id)], queue="build")
    return {"status": "queued"}
