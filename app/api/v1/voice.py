"""ElevenLabs Conversational AI — voice chat for projects.

Two endpoints:

  GET  /api/v1/voice/session/{project_id}
       Returns a short-lived signed ElevenLabs WebSocket URL the frontend
       connects to directly. The project context (name, framework, system
       prompt) and a HMAC signature are embedded so our /llm endpoint can
       verify the session without a separate token store.

  POST /api/v1/voice/llm
       OpenAI-compatible streaming endpoint called server-to-server by
       ElevenLabs when the user speaks. Runs our chat classifier and either
       returns a spoken reply or queues an update — same logic as /chat but
       tuned for brevity (voice responses must be 1-2 sentences).

ElevenLabs agent setup (one-time, in the ElevenLabs dashboard):
  1. Create an agent → LLM → Custom
  2. Set the Custom LLM URL to:  https://<your-domain>/api/v1/voice/llm
  3. Choose a voice (e.g. Rachel / Aria)
  4. Copy the Agent ID into ELEVENLABS_AGENT_ID in your .env
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import logging
import re as _re
import uuid

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.exceptions import ForbiddenError, NotFoundError
from app.deps import CurrentUser, DBSession, SettingsDep

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sign(secret: str, project_id: str, user_id: str) -> str:
    payload = f"{project_id}:{user_id}".encode()
    return _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _sign_assistant(secret: str, user_id: str) -> str:
    payload = f"assistant:{user_id}".encode()
    return _hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _fw_label(template_key: str) -> str:
    return {"flutter": "Flutter", "next": "Next.js", "react_native": "React Native"}.get(
        template_key, template_key
    )


def _sse(text: str) -> StreamingResponse:
    """Wrap a single text reply as an OpenAI streaming SSE response."""

    async def gen():
        chunk = json.dumps({
            "choices": [{"delta": {"role": "assistant", "content": text}, "index": 0, "finish_reason": None}]
        })
        yield f"data: {chunk}\n\n"
        done = json.dumps({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]})
        yield f"data: {done}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Signed session URL ────────────────────────────────────────────────────────


@router.get("/session/{project_id}")
async def get_voice_session(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
    settings: SettingsDep,
) -> dict:
    """Mint a signed ElevenLabs WebSocket URL for the given project.

    The frontend connects to the returned URL directly — no audio ever
    flows through our backend.
    """
    if not settings.ELEVENLABS_API_KEY or not settings.ELEVENLABS_AGENT_ID:
        raise HTTPException(status_code=503, detail="Voice chat is not configured on this server.")

    from app.api.v1.projects import _get_owned

    project = await _get_owned(project_id, user.id, db)
    fw = _fw_label(project.template_key)
    sig = _sign(settings.ELEVENLABS_VOICE_SECRET, str(project_id), str(user.id))

    system_prompt = (
        f'You are the Forgefy AI assistant for "{project.app_name}", a {fw} app.\n\n'
        "Help the user build and improve their app through natural voice conversation.\n"
        "Keep ALL responses short and conversational — this is voice, not text.\n"
        "When the user asks for a change, give a brief confirmation like "
        '"Got it, I\'ll add that now" — never list steps or ask follow-up questions '
        "unless the request is completely unclear."
    )

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url",
            headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
            json={
                "agent_id": settings.ELEVENLABS_AGENT_ID,
                "conversation_config_override": {
                    "agent": {
                        "prompt": {
                            "prompt": system_prompt,
                            "custom_llm_extra_body": {
                                "project_id": str(project_id),
                                "user_id": str(user.id),
                                "sig": sig,
                            },
                        },
                        "first_message": (
                            f"Hi! I'm your Forgefy assistant for {project.app_name}. "
                            "What would you like to build or change?"
                        ),
                    }
                },
            },
        )

    if not resp.is_success:
        logger.error("ElevenLabs signed URL failed %s: %s", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail="Could not start voice session.")

    return {"signed_url": resp.json()["signed_url"]}


@router.get("/assistant-session")
async def get_assistant_voice_session(
    user: CurrentUser,
    settings: SettingsDep,
) -> dict:
    """Mint a signed ElevenLabs WebSocket URL for the global assistant (no project context).

    The assistant agent uses the same ElevenLabs agent as projects but with a
    different system prompt and mode="assistant" in the custom body, so the
    /voice/llm endpoint routes to the assistant AI instead of the project AI.
    """
    if not settings.ELEVENLABS_API_KEY or not settings.ELEVENLABS_AGENT_ID:
        raise HTTPException(status_code=503, detail="Voice chat is not configured on this server.")

    sig = _sign_assistant(settings.ELEVENLABS_VOICE_SECRET, str(user.id))

    system_prompt = (
        "You are the Forgefy voice assistant. Voice-first: all responses MUST be 1-2 short sentences.\n"
        "Help users with the Forgefy platform — turning meeting recordings into Flutter, React Native, "
        "and Next.js apps. Guide them through sessions, projects, blueprints, and the dashboard.\n"
        "Never use markdown, bullet points, or code blocks. Speak naturally."
    )

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url",
            headers={"xi-api-key": settings.ELEVENLABS_API_KEY},
            json={
                "agent_id": settings.ELEVENLABS_AGENT_ID,
                "conversation_config_override": {
                    "agent": {
                        "prompt": {
                            "prompt": system_prompt,
                            "custom_llm_extra_body": {
                                "user_id": str(user.id),
                                "sig": sig,
                                "mode": "assistant",
                            },
                        },
                        "first_message": "Hi! I'm your Forgefy assistant. What can I help you with?",
                    }
                },
            },
        )

    if not resp.is_success:
        logger.error("ElevenLabs assistant session failed %s: %s", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail="Could not start voice session.")

    return {"signed_url": resp.json()["signed_url"]}


# ── Custom LLM (called server-to-server by ElevenLabs) ───────────────────────


@router.post("/llm")
async def voice_llm(
    request: Request,
    db: DBSession,
    settings: SettingsDep,
) -> StreamingResponse:
    """OpenAI-compatible streaming endpoint called by ElevenLabs when the user speaks.

    ElevenLabs passes our custom_llm_extra_body fields (project_id, user_id,
    sig) at the top level of the request body alongside the standard OpenAI
    fields (model, messages, stream).
    """
    body = await request.json()

    # ── Assistant mode (global assistant, no project context) ─────────────────
    if body.get("mode") == "assistant":
        user_id_str = body.get("user_id", "")
        sig = body.get("sig", "")

        if settings.ELEVENLABS_VOICE_SECRET:
            expected = _sign_assistant(settings.ELEVENLABS_VOICE_SECRET, user_id_str)
            if not _hmac.compare_digest(sig, expected):
                raise HTTPException(status_code=401, detail="Invalid voice session signature")

        messages = body.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        message = user_msgs[-1].get("content", "").strip() if user_msgs else ""

        if not message:
            return _sse("I didn't catch that. What can I help you with?")

        system = (
            "You are the Forgefy assistant. Voice-first: keep every reply to 1-2 sentences.\n"
            "Forgefy joins planning meetings and builds Flutter, React Native, and Next.js apps from them.\n"
            "Help users with sessions, projects, blueprints, and the dashboard.\n"
            "Never use markdown, bullet points, or code formatting."
        )
        try:
            reply = await _call_assistant_llm(system, message, settings)
        except Exception as exc:
            logger.error("Voice assistant LLM error: %s", exc, exc_info=True)
            reply = "I'm having a moment. Please try again."

        return _sse(reply)

    # ── Project mode ──────────────────────────────────────────────────────────
    project_id_str = body.get("project_id", "")
    user_id_str = body.get("user_id", "")
    sig = body.get("sig", "")

    # Verify HMAC — rejects calls that didn't originate from a signed session
    if settings.ELEVENLABS_VOICE_SECRET:
        expected = _sign(settings.ELEVENLABS_VOICE_SECRET, project_id_str, user_id_str)
        if not _hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=401, detail="Invalid voice session signature")

    # Extract the last user turn from the conversation history
    messages = body.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    message = user_messages[-1].get("content", "").strip() if user_messages else ""

    if not message:
        return _sse("I didn't catch that. What would you like to do?")

    try:
        project_id = uuid.UUID(project_id_str)
        user_id = uuid.UUID(user_id_str)
    except (ValueError, AttributeError):
        return _sse("Sorry, I lost track of which project we're working on.")

    try:
        from app.api.v1.projects import _get_owned

        project = await _get_owned(project_id, user_id, db)
    except (NotFoundError, ForbiddenError):
        return _sse("I couldn't find that project.")

    fw = _fw_label(project.template_key)
    reply, should_update, update_prompt = await _classify(message, project, fw, settings)

    if should_update and update_prompt and not project.is_updating:
        try:
            from app.core.dispatch import dispatch
            from app.workers.update_worker import apply_update

            await dispatch(
                apply_update,
                args=[str(project_id), update_prompt, str(user_id)],
                queue="build",
            )
            logger.info("Voice update queued project=%s prompt_len=%d", project_id, len(update_prompt))
        except Exception as exc:
            logger.error("Voice update dispatch failed project=%s: %s", project_id, exc)

    return _sse(reply)


# ── LLM classification ────────────────────────────────────────────────────────


async def _classify(
    message: str,
    project,
    fw: str,
    settings,
) -> tuple[str, bool, str]:
    """Classify a voice message into (spoken_reply, should_update, update_prompt)."""
    system = (
        f'You are the Forgefy AI assistant for "{project.app_name}", a {fw} app.\n\n'
        "The user is speaking via voice — keep responses 1-2 sentences, natural for speech.\n\n"
        "Reply ONLY with valid JSON (no markdown, no extra text):\n"
        "{\n"
        '  "type": "chat" | "update",\n'
        '  "response": "<short spoken reply, 1-2 sentences>",\n'
        '  "update_prompt": "<detailed technical instruction — only when type is update>"\n'
        "}\n\n"
        'Use "update" for any request to add, change, fix, or improve the app.\n'
        'Use "chat" for greetings, questions about the app, status checks.\n'
        "For update: keep the spoken reply brief (\"Got it, I'll do that now\") but make "
        "update_prompt a full technical spec the build agent can act on."
    )

    try:
        raw = await _call_llm(system, message, settings)
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        m = _re.search(r"\{.*\}", raw, flags=_re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            return (
                parsed.get("response", "Got it."),
                parsed.get("type") == "update",
                parsed.get("update_prompt", "").strip(),
            )
    except Exception as exc:
        logger.error("Voice classifier error: %s", exc, exc_info=True)

    return "I'm having trouble with that right now. Please try again.", False, ""


async def _call_llm(system: str, message: str, settings) -> str:
    """Call the configured LLM synchronously in a thread and return raw text."""
    import asyncio

    if settings.BUILD_MODEL == "gemini":
        import requests as _req

        def _gemini() -> str:
            r = _req.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent",
                params={"key": settings.GEMINI_API_KEY},
                json={
                    "system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": message}]}],
                    "generationConfig": {"maxOutputTokens": 512},
                },
                timeout=30,
            )
            r.raise_for_status()
            parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()

        return await asyncio.to_thread(_gemini)

    if settings.BUILD_MODEL == "Qwen3":
        import requests as _req

        def _qwen() -> str:
            from app.ai.qwen import using_openrouter

            if using_openrouter():
                from app.ai.openrouter import PLAN, chat_openrouter

                return chat_openrouter(system, message, task=PLAN, max_tokens=512)
            from app.ai.ollama_http import ollama_base_url, ollama_headers

            r = _req.post(
                f"{ollama_base_url(settings)}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": message},
                    ],
                    "stream": False,
                },
                headers=ollama_headers(settings),
                timeout=60,
            )
            r.raise_for_status()
            return (r.json().get("message") or {}).get("content", "").strip()

        return await asyncio.to_thread(_qwen)

    # Default: Anthropic Claude
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": message}],
    )
    return resp.content[0].text.strip() if resp.content else ""


async def _call_assistant_llm(system: str, message: str, settings) -> str:
    """Call the assistant-role model — same AI the text assistant uses.

    Prefers the OpenRouter ASSISTANT chain (fast conversational model); falls
    back to the generic LLM when OpenRouter isn't configured.
    """
    import asyncio

    if (settings.OPENROUTER_API_KEY or "").strip():
        from app.ai.openrouter import ASSISTANT, chat_openrouter

        return await asyncio.to_thread(
            lambda: chat_openrouter(system, message, task=ASSISTANT, max_tokens=256, json_mode=False)
        )
    return await _call_llm(system, message, settings)
