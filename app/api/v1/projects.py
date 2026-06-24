"""Projects endpoints — list, get, and dispatch prompt-driven updates."""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter

from app.core.exceptions import ForbiddenError, NotFoundError, ValidationError
from app.deps import CurrentUser, DBSession
from app.schemas.project import ChatHistoryRequest, ChatRequest, ChatResponse, ProjectOut, UpdateProjectRequest

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
    )


async def _get_owned(project_id: uuid.UUID, user_id: uuid.UUID, db) -> ProjectOut:
    doc = await db.collection("projects").document(str(project_id)).get()
    if not doc.exists:
        raise NotFoundError(f"Project {project_id} not found")
    d = doc.to_dict()
    if uuid.UUID(d["owner_id"]) != user_id:
        raise ForbiddenError("Access denied")
    return _doc_to_out(doc)


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


@router.post("/{project_id}/update", response_model=dict)
async def update_project(
    project_id: uuid.UUID,
    body: UpdateProjectRequest,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Dispatch a prompt-driven update for the project. Returns immediately."""
    project = await _get_owned(project_id, user.id, db)

    if project.is_updating:
        raise ValidationError("A build or update is already in progress.")

    from app.workers.update_worker import apply_update
    apply_update.apply_async(
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
    project = await _get_owned(project_id, user.id, db)
    message = body.message.strip()

    if not message:
        return ChatResponse(type="chat", response="I didn't catch that — what would you like to do?")

    from app.config import get_settings

    settings = get_settings()
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
        apply_update.apply_async(
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

3. "update"
   Use for ANY request that names features, screens, behaviour, or improvements —
   no matter how large or how many items are listed. When in doubt, choose "update".
   response: a short, enthusiastic confirmation of what will be built.
   update_prompt: pass the user's request through with full detail, expanding
   any implicit requirements so the planner has everything it needs.

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
  "update_prompt": "<detailed technical instruction — only present when type is update>"
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
        return ChatResponse(type="chat", response="I'm here to help — what would you like to do?")

    intent = parsed.get("type", "chat")
    user_response = parsed.get("response", "")
    update_prompt = parsed.get("update_prompt", "").strip()

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
        apply_update.apply_async(
            args=[str(project_id), update_prompt, str(user.id)],
            queue="build",
        )
        logger.info("Queued update project=%s prompt_len=%d", project_id, len(update_prompt))
        return ChatResponse(type="update", response=user_response, update_queued=True)

    # "chat" or "clarify" — just return the response, nothing queued
    return ChatResponse(type=intent, response=user_response)


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
    from datetime import datetime, timezone
    messages = body.messages[-100:]
    await db.collection("project_chats").document(str(project_id)).set(
        {"messages": messages, "updated_at": datetime.now(timezone.utc)},
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
    build_preview.apply_async(
        args=[str(project_id), str(user.id)],
        queue="build",
    )
    return {"status": "queued"}
