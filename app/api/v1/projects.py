"""Projects endpoints — list, get, and dispatch prompt-driven updates."""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter

from app.core.exceptions import ForbiddenError, NotFoundError, ValidationError
from app.deps import CurrentUser, DBSession
from app.schemas.project import ChatRequest, ChatResponse, ProjectOut, UpdateProjectRequest

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

    system = f"""You are the Forgefy AI assistant for "{project.app_name}", a {fw} app.

App context (what already exists):
{json.dumps(blueprint_summary, indent=2)}

──────────────────────────────────────────────
YOUR JOB
──────────────────────────────────────────────
Analyse the user's message and pick ONE of three response types:

1. "chat"
   Use for: greetings, thanks, questions about the app, general conversation.
   response: a short, friendly reply.

2. "clarify"
   Use for: requests that are too vague to implement without guessing.
   Examples: "make it better", "fix the app", "add something cool"
   response: ask ONE specific question that will let you generate a precise instruction.

3. "update"
   Use for: any clear request to add, change, fix, or remove something in the app.
   response: a short, friendly confirmation of what you will do.
   update_prompt: a DETAILED, step-by-step technical instruction for the build agent.

──────────────────────────────────────────────
HOW TO WRITE A GOOD update_prompt
──────────────────────────────────────────────
The build agent reads files and writes code — it needs precision.

Bad:  "add animations"
Good: "Add entrance animations to the home screen and the list items in the dashboard screen.
       In Flutter use AnimatedOpacity + SlideTransition triggered on initState.
       Animate each list item with a staggered delay (50ms per item)."

Bad:  "add onboarding"
Good: "Create a 3-screen onboarding flow (Welcome, Features, Get Started).
       In Flutter: create lib/features/onboarding/presentation/pages/onboarding_page.dart
       using a PageView with three OnboardingStep widgets (icon, title, body, skip/next buttons).
       Register it as the initial route in lib/app.dart only when onboarding_complete is not
       set in SharedPreferences."

Always reference the actual features/screens from the blueprint above.
Always name the files/classes to create or modify.
Always specify framework-specific APIs (Flutter widgets, React hooks, Next.js patterns).

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
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            ai_resp = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": message}],
            )
            raw = ai_resp.content[0].text.strip() if ai_resp.content else ""
    except Exception as exc:
        logger.warning("Chat classifier failed: %s", exc)
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
