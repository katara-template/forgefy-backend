"""Global dashboard help assistant.

A conversational guide for Forgefy itself — understanding sessions, projects,
token usage, "how do I…" questions, and finding one's way around — as opposed to
chat_with_project in projects.py, which edits a single generated app.

Signed-in users get multiple named conversation threads: each thread carries its
own message history (its context), while learned memory — durable facts about the
user — is shared across all of their threads. Anonymous visitors get a single,
non-persisted thread (recent turns are supplied by the client per request).

Storage (Firestore):
- user_assistant/{uid}.memory        → per-user durable facts (all threads)
- assistant_conversations/{cid}      → one thread: owner_id, title, messages, timestamps
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter

from app.ai.openrouter import ASSISTANT, OpenRouterError, call_openrouter
from app.core.dispatch import dispatch
from app.core.exceptions import (
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)
from app.deps import DBSession, OptionalUser
from app.schemas.assistant import (
    AssistantAction,
    AssistantBuildRequest,
    AssistantBuildResponse,
    AssistantChatRequest,
    AssistantChatResponse,
    AssistantLink,
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_CONV_COLLECTION = "assistant_conversations"

# Routes the assistant is allowed to deep-link to. A hallucinated or off-site
# link would send the user somewhere broken (or worse, off Forgefy), so we only
# surface links whose first path segment is a real route. Anonymous-facing
# routes (/login, /register) are included so it can point visitors at sign-up.
_ALLOWED_LINK_ROOTS = frozenset(
    {
        "dashboard",
        "sessions",
        "projects",
        "developers",
        "billing",
        "settings",
        "documentation",
        "login",
        "register",
    }
)

# Session platforms the start_session action may carry (mirrors the New Session
# form on /sessions). Anything else is dropped so the UI never gets a bad value.
_VALID_PLATFORMS = frozenset({"meet", "zoom", "teams", "physical"})

# Actions the model may request. Anything else collapses to "none".
_VALID_ACTIONS = frozenset({"none", "start_session", "build_app", "auth_required"})

# How much state to carry. History is capped so each thread doc stays small and
# the prompt bounded; memory is the distilled long-term layer.
_HISTORY_TURNS = 8          # last N messages fed back into the prompt
_HISTORY_CAP = 100          # messages retained per thread
_MEMORY_CAP = 25            # durable facts retained per user
_DEFAULT_TITLE = "New chat"

_FRAMEWORKS = {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}


# ── Storage helpers ──────────────────────────────────────────────────────────

def _memory_ref(db, user_id: str):
    return db.collection("user_assistant").document(user_id)


def _conv_ref(db, conversation_id: str):
    return db.collection(_CONV_COLLECTION).document(conversation_id)


async def _load_memory(db, user_id: str) -> list[str]:
    doc = await _memory_ref(db, user_id).get()
    if not doc.exists:
        return []
    return (doc.to_dict() or {}).get("memory") or []


async def _owned_conversation(db, conversation_id: str, user_id: str) -> dict:
    """Load a thread and assert the caller owns it. Raises otherwise."""
    doc = await _conv_ref(db, conversation_id).get()
    if not doc.exists:
        raise NotFoundError("Conversation not found")
    data = doc.to_dict() or {}
    if data.get("owner_id") != user_id:
        raise ForbiddenError("Access denied")
    return {"id": doc.id, **data}


def _title_from(message: str) -> str:
    text = " ".join(message.split()).strip()
    return (text[:48] or _DEFAULT_TITLE) if text else _DEFAULT_TITLE


# ── Context building ─────────────────────────────────────────────────────────

async def _gather_context(db, user_id: str) -> str:
    """A compact, model-readable snapshot of the signed-in user's account.

    Best-effort: a failure to read any slice must never break the chat, so the
    whole thing is guarded and simply yields less context on error. Each project
    carries its own route so the assistant can point the user straight at it.
    """
    lines: list[str] = []

    try:
        docs = await db.collection("projects").where("owner_id", "==", user_id).get()
        projects = []
        for d in docs:
            p = d.to_dict() or {}
            fw = _FRAMEWORKS.get(p.get("template_key", ""), p.get("template_key", "app"))
            status = (
                "building/updating"
                if p.get("is_updating")
                else ("build error" if p.get("build_error") else "ready")
            )
            projects.append(
                {
                    "name": p.get("app_name", "untitled"),
                    "framework": fw,
                    "status": status,
                    "has_preview": bool(p.get("preview_url")),
                    "route": f"/projects/{d.id}",
                }
            )
        lines.append(f"Projects ({len(projects)}): {json.dumps(projects)}")
    except Exception:  # noqa: BLE001 — context is optional, never fatal
        logger.debug("assistant: could not load projects for %s", user_id, exc_info=True)

    try:
        from app.core import usage
        from app.core.tiers import get_tier

        tier_key = await usage.get_user_tier(db, user_id)
        tier = get_tier(tier_key)
        used = await usage.get_monthly_tokens(db, user_id)
        remaining = max(0, tier.monthly_tokens - used)
        lines.append(
            f"Plan: {tier.name} ({tier_key}). Tokens this month: {used} used, "
            f"{remaining} of {tier.monthly_tokens} remaining. Resets {usage.quota_reset_label()}."
        )
    except Exception:  # noqa: BLE001
        logger.debug("assistant: could not load usage for %s", user_id, exc_info=True)

    return "\n".join(lines) if lines else "(no account data available)"


def _recent_history(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages[-_HISTORY_TURNS:]:
        role = m.get("role", "user")
        text = (m.get("text") or "").strip()[:400]
        if not text or role == "error":
            continue
        lines.append(f'{"User" if role == "user" else "You"}: {text}')
    return "\n".join(lines)


def _build_system_prompt(
    *,
    authed: bool,
    context: str,
    memory: list[str],
    history: str,
    page: str | None,
    build_mode: bool = False,
) -> str:
    page_block = f'\nThe user is currently on the "{page}" page.' if page else ""
    history_block = f"\n\nThis thread so far:\n{history}" if history else ""
    build_block = (
        """

BUILD MODE IS ON
The user wants to BUILD an app. Treat their message as an app idea.
- If essential details are missing (what the app does, its main features, or who
  it's for), ask ONE short, friendly clarifying question and keep action.type
  "none". Ask at most one or two questions total — don't interrogate.
- As soon as you have a clear picture, set action.type "build_app" with a refined
  1-3 sentence description, and say in "response" that you're starting the build.
- Lean toward building: a single clear sentence of intent is enough to start."""
        if build_mode and authed
        else ""
    )

    if authed:
        account_block = f"""THE SIGNED-IN USER'S ACCOUNT (ground answers in this — be specific, not generic):
{context}

WHAT YOU REMEMBER ABOUT THIS USER (from past conversations):
{(chr(10).join(f'- {f}' for f in memory)) if memory else '(nothing remembered yet)'}"""
        auth_rules = """CAPABILITIES — STARTING A SESSION
- You can START A SESSION yourself (Forgefy joining a meeting to build an app).
  You do this directly — never send the user to a form to do it.
- To start one you need:
    • platform: one of "meet" (Google Meet), "zoom", "teams", or "physical"
      (an in-person meeting recorded via mic/upload).
    • meeting_url: REQUIRED for meet/zoom/teams; NOT needed for physical.
- If the user wants to start a session but hasn't given the platform (and the
  meeting link, for online platforms), ASK for the missing piece in "response" —
  one short, friendly question — and keep action.type = "none". Never invent a
  meeting URL.
- Once you have everything, set action.type = "start_session" with
  action.platform and action.meeting_url filled in. The app creates it, joins the
  meeting, and opens it; your "response" should say you're starting it.

CAPABILITIES — BUILDING AN APP FROM A PROMPT
- You can BUILD A FULL APP directly from the user's idea — no meeting needed.
  Forgefy generates a blueprint (features + design) and builds it automatically.
- When the user asks you to build/create/make an app, set action.type = "build_app"
  and put a clear 1-3 sentence spec of the app in action.description (what it
  does and who it's for). In "response", say you're starting the build.
- If the idea is too vague to build (e.g. "build me something cool"), ask ONE
  short question to pin down what the app should do, and keep action.type "none".
  Don't over-interrogate — a single clear sentence of intent is enough to start.
- To point the user at one of their apps, link to that project's own route (the
  "route" field in the account data above)."""
    else:
        account_block = """THE VISITOR IS NOT SIGNED IN.
They have no projects, usage, or history yet. You can still explain how Forgefy
works and help them find their way around."""
        auth_rules = """CAPABILITIES & AUTH
- The visitor is anonymous. Anything that needs an account — starting a session,
  seeing their projects/sessions/billing, or saving anything — REQUIRES sign-in.
- When they ask for something like that, set action.type = "auth_required" and, in
  "response", warmly invite them to sign in or create a free account. Sign-in
  methods available: Google, GitHub, or email + password. The app will show the
  sign-in options right in this chat, so you don't need to collect credentials.
- For purely informational or navigational questions, just answer (action "none").
  You may link to /login or /register when it helps."""

    return f"""You are the Forgefy assistant — a friendly, concise in-app guide.

WHAT FORGEFY IS
Forgefy joins a team's planning calls, extracts what they actually decided, and
builds Flutter, React Native, and Next.js apps from it — simultaneously.

THE DASHBOARD (use these exact routes when you link somewhere)
- /dashboard — overview and getting started
- /sessions — meeting/recording sessions that get turned into apps; new sessions start here
- /projects — the generated apps; each can be previewed and refined by chatting inside it
- /developers — API keys and SDK for the developer API
- /billing — plan, monthly token budget, and upgrades
- /settings — account settings
- /login, /register — sign in or create an account

{account_block}{page_block}{history_block}

{auth_rules}{build_block}

YOUR JOB
Answer the question or guide the user to their next step. Keep replies short
(1-3 sentences) and warm. When a page helps, point them to it with a link.

OUTPUT — reply with ONLY a JSON object, no prose around it:
{{
  "response": "your reply to show the user (plain text or light markdown)",
  "links": [{{"label": "Open Billing", "to": "/billing"}}],
  "action": {{"type": "none", "platform": null, "meeting_url": null, "description": null}},
  "remember": ["a durable fact worth recalling next time"]
}}

RULES
- "links": 0-3 items. "to" MUST be one of the routes listed above. No external URLs.
- "action.type": one of "none", "start_session", "build_app", "auth_required". Use "none" unless a real action is intended.
- For "build_app", put the app spec in "action.description".
- "remember": only when the user reveals something durable about their goals, preferences, or projects; else []. Never store secrets, tokens, or passwords.
- If you don't know something, say so plainly rather than inventing details."""


# ── Response sanitisation ────────────────────────────────────────────────────

def _sanitize_links(raw: object) -> list[AssistantLink]:
    if not isinstance(raw, list):
        return []
    out: list[AssistantLink] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        to = str(item.get("to", "")).strip()
        label = str(item.get("label", "")).strip()
        if not to.startswith("/") or not label:
            continue
        root = to.strip("/").split("/", 1)[0]
        if root not in _ALLOWED_LINK_ROOTS or to in seen:
            continue
        seen.add(to)
        out.append(AssistantLink(label=label[:40], to=to))
        if len(out) >= 3:
            break
    return out


def _parse_action(raw: object, *, authed: bool) -> AssistantAction:
    """Validate the model's action, enforce the auth gate, and require that a
    start_session is complete enough to create outright."""
    action = AssistantAction()
    if isinstance(raw, dict):
        atype = str(raw.get("type", "none")).strip()
        if atype in _VALID_ACTIONS:
            action.type = atype
        platform = str(raw.get("platform") or "").strip().lower()
        if platform in _VALID_PLATFORMS:
            action.platform = platform
        url = str(raw.get("meeting_url") or "").strip()
        if url:
            action.meeting_url = url[:500]
        desc = str(raw.get("description") or "").strip()
        if desc:
            action.description = desc[:2000]

    if not authed and action.type != "none":
        # Privileged intent from an anonymous visitor → gate behind sign-in.
        return AssistantAction(type="auth_required")

    # Completeness gate: only surface start_session when the app can create it
    # outright (platform, plus a URL for online meetings). If details are still
    # missing the assistant should have asked for them, so drop to plain advice
    # rather than kicking off a broken create or bouncing the user to a form.
    if action.type == "start_session":
        if not action.platform or (
            action.platform != "physical" and not action.meeting_url
        ):
            action.type = "none"
    # A build needs a real spec — without one the assistant should have asked.
    elif action.type == "build_app" and not (action.description and len(action.description) >= 8):
        action.type = "none"
    return action


def _merge_memory(existing: list[str], raw_new: object) -> list[str]:
    """Append new facts, de-duplicated case-insensitively, newest-capped."""
    merged = list(existing)
    lowered = {f.lower() for f in merged}
    if isinstance(raw_new, list):
        for fact in raw_new:
            text = str(fact).strip()
            if text and text.lower() not in lowered:
                merged.append(text[:200])
                lowered.add(text.lower())
    return merged[-_MEMORY_CAP:]


def _sanitize_client_history(raw: list[dict] | None) -> list[dict]:
    """Coerce anonymous visitors' client-sent history into safe {role, text}."""
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for m in raw[-_HISTORY_TURNS:]:
        if not isinstance(m, dict):
            continue
        role = "user" if m.get("role") == "user" else "assistant"
        text = str(m.get("text") or "").strip()
        if text:
            out.append({"role": role, "text": text})
    return out


# ── Endpoints ────────────────────────────────────────────────────────────────

async def process_chat(db, user, body: AssistantChatRequest) -> AssistantChatResponse:
    """Core chat logic shared by the HTTP endpoint and the WebSocket channel.

    `user` is a User or None (anonymous). Anonymous callers are stateless.
    """
    authed = user is not None
    message = body.message.strip()
    if not message:
        return AssistantChatResponse(response="What can I help you with?", authenticated=authed)

    conv_messages: list[dict] = []
    conv_title: str | None = None
    cid = (body.conversation_id or "").strip() or None

    if user is not None:
        from app.core.usage import check_not_over_limit

        uid = str(user.id)
        await check_not_over_limit(db, uid)
        memory = await _load_memory(db, uid)
        if cid is not None:
            conv = await _owned_conversation(db, cid, uid)  # raises if not owned
            conv_messages = conv.get("messages") or []
            conv_title = conv.get("title")
        context = await _gather_context(db, uid)
        history = _recent_history(conv_messages)
    else:
        uid = None
        memory = []
        context = ""
        history = _recent_history(_sanitize_client_history(body.history))

    system = _build_system_prompt(
        authed=authed,
        context=context,
        memory=memory,
        history=history,
        page=body.page,
        build_mode=(body.mode == "build"),
    )

    try:
        parsed = call_openrouter(system, message, task=ASSISTANT, max_tokens=1200)
    except OpenRouterError as exc:
        logger.warning("assistant chat failed (authed=%s): %s", authed, exc)
        return AssistantChatResponse(
            response="I'm having trouble thinking right now — please try again in a moment.",
            authenticated=authed,
            conversation_id=cid,
        )

    reply = str(parsed.get("response") or "").strip()
    if not reply:
        reply = "Sorry, I didn't quite get that — could you rephrase?"
    links = _sanitize_links(parsed.get("links"))
    action = _parse_action(parsed.get("action"), authed=authed)

    # Persist for signed-in users: append to the thread (creating it if new) and
    # fold any learned facts into shared memory.
    if uid is not None:
        now = datetime.now(UTC)
        new_messages = (
            conv_messages + [{"role": "user", "text": message}, {"role": "assistant", "text": reply}]
        )[-_HISTORY_CAP:]

        if cid is None:
            cid = str(uuid.uuid4())
            await _conv_ref(db, cid).set(
                {
                    "owner_id": uid,
                    "title": _title_from(message),
                    "messages": new_messages,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        else:
            update: dict = {"messages": new_messages, "updated_at": now}
            if not conv_title or conv_title == _DEFAULT_TITLE:
                update["title"] = _title_from(message)
            await _conv_ref(db, cid).set(update, merge=True)

        memory = _merge_memory(memory, parsed.get("remember"))
        await _memory_ref(db, uid).set({"memory": memory}, merge=True)

    return AssistantChatResponse(
        response=reply,
        links=links,
        action=action if action.type != "none" else None,
        authenticated=authed,
        conversation_id=cid,
    )


@router.post("/chat", response_model=AssistantChatResponse)
async def assistant_chat(
    body: AssistantChatRequest,
    db: DBSession,
    user: OptionalUser,
) -> AssistantChatResponse:
    """Answer a help question within a thread (HTTP). See also /ws/assistant/chat."""
    return await process_chat(db, user, body)


@router.post("/build", response_model=AssistantBuildResponse)
async def assistant_build(
    body: AssistantBuildRequest,
    db: DBSession,
    user: OptionalUser,
) -> AssistantBuildResponse:
    """Start a prompt → blueprint → build run; returns the new project's id.

    Creates a prompt-originated session (already in PROCESSING) plus a project
    stub the UI can open immediately, then hands off to the prompt build worker,
    which designs the blueprint and dispatches the normal build pipeline.
    """
    if user is None:
        raise UnauthorizedError("Sign in to build an app")

    description = body.description.strip()
    if len(description) < 8:
        raise ValidationError("Please describe the app you'd like to build in a sentence or two.")

    from app.core.usage import check_not_over_limit
    from app.db.models.enums import Platform, SessionStatus

    uid = str(user.id)
    await check_not_over_limit(db, uid)

    now = datetime.now(UTC)
    session_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())

    await db.collection("sessions").document(session_id).set({
        "user_id": uid,
        "status": SessionStatus.PROCESSING.value,
        "platform": Platform.PHYSICAL.value,
        "meeting_url": None,
        "origin": "prompt",
        "prompt": description[:2000],
        "start_time": None,
        "end_time": None,
        "created_at": now,
    })

    stub_name = re.sub(r"[^a-zA-Z0-9._-]", "-", description[:40]).strip("-").lower() or "new-app"
    await db.collection("projects").document(project_id).set({
        "owner_id": uid,
        "session_id": session_id,
        "blueprint_id": None,
        "app_name": stub_name,
        "template_key": "next",
        "repo_full_name": "",
        "github_url": "",
        "preview_url": None,
        "artifact_url": None,
        "is_updating": True,
        "build_error": None,
        "blueprint_context": {"app_description": description},
        "created_at": now,
        "updated_at": now,
    })

    from app.workers.prompt_build_worker import build_from_prompt
    await dispatch(build_from_prompt, args=[session_id, project_id, description, uid], queue="build")
    logger.info("Prompt build queued user=%s project=%s", uid, project_id)

    return AssistantBuildResponse(project_id=project_id, session_id=session_id)


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(db: DBSession, user: OptionalUser) -> ConversationListResponse:
    """List the user's threads, most-recently-updated first. Empty for anonymous."""
    if user is None:
        return ConversationListResponse()
    docs = await db.collection(_CONV_COLLECTION).where("owner_id", "==", str(user.id)).get()
    summaries = [
        ConversationSummary(
            id=d.id,
            title=(d.to_dict() or {}).get("title") or _DEFAULT_TITLE,
            created_at=(d.to_dict() or {}).get("created_at"),
            updated_at=(d.to_dict() or {}).get("updated_at"),
        )
        for d in docs
    ]
    _fallback = datetime.min.replace(tzinfo=UTC)
    summaries.sort(key=lambda c: c.updated_at or _fallback, reverse=True)
    return ConversationListResponse(conversations=summaries)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str, db: DBSession, user: OptionalUser
) -> ConversationDetail:
    """Return a thread's messages so the widget can switch into it."""
    if user is None:
        raise UnauthorizedError("Sign in to view conversations")
    conv = await _owned_conversation(db, conversation_id, str(user.id))
    return ConversationDetail(
        id=conv["id"],
        title=conv.get("title") or _DEFAULT_TITLE,
        messages=conv.get("messages") or [],
    )


@router.delete("/conversations/{conversation_id}", response_model=dict)
async def delete_conversation(
    conversation_id: str, db: DBSession, user: OptionalUser
) -> dict:
    """Delete a thread. Learned memory (shared across threads) is preserved."""
    if user is None:
        raise UnauthorizedError("Sign in to manage conversations")
    await _owned_conversation(db, conversation_id, str(user.id))  # ownership check
    await _conv_ref(db, conversation_id).delete()
    return {"deleted": True}


