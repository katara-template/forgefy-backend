"""Update worker — applies a user prompt to an existing project."""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _load_project(project_id: str) -> dict:
    from app.db.firebase import get_firestore_client

    db = get_firestore_client()
    doc = await db.collection("projects").document(project_id).get()
    if not doc.exists:
        raise ValueError(f"Project {project_id} not found")
    return doc.to_dict() | {"id": doc.id}


async def _patch_project(project_id: str, updates: dict) -> None:
    from app.db.firebase import get_firestore_client

    db = get_firestore_client()
    await db.collection("projects").document(project_id).update(updates)




async def _run(project_id: str, prompt: str, user_id: str) -> dict:
    from app.build.workspace import EditWorkspace
    from app.build.build_agent import run_update_agent
    from app.db.firebase import refresh_async_firestore_client
    from app.core.usage import get_user_tier, is_over_limit, record_usage

    settings = get_settings()
    db = refresh_async_firestore_client()  # bind fresh gRPC channel to this event loop

    project = await _load_project(project_id)
    app_name: str = project["app_name"]
    repo_full_name: str = project["repo_full_name"]
    template_key: str = project.get("template_key", "")
    blueprint_context: dict = project.get("blueprint_context") or {}

    # Get the right GitHub token (validates personal token; falls back to system if invalid)
    from app.build.github_token import get_valid_github_token
    github_token = await get_valid_github_token(user_id, settings.GITHUB_TOKEN)
    push_url = f"https://{github_token}@github.com/{repo_full_name}.git"

    # Check token quota before doing any expensive work
    user_tier = await get_user_tier(db, user_id)
    if await is_over_limit(db, user_id, user_tier):
        from app.core.tiers import get_tier
        from app.build.build_logger import publish_user_event
        tier = get_tier(user_tier)
        error_msg = (
            f"You've used all {tier.monthly_tokens:,} tokens in your {tier.name} plan this month. "
            "Upgrade to continue."
        )
        await _patch_project(project_id, {
            "build_error": error_msg,
            "build_error_action": "support",
            "updated_at": datetime.now(timezone.utc),
        })
        publish_user_event(
            settings.REDIS_URL, user_id, "quota_exceeded", error_msg,
            tier=user_tier, limit=tier.monthly_tokens,
        )
        logger.warning("Update blocked — token limit reached user=%s tier=%s", user_id, user_tier)
        return {"blocked": "token_limit"}

    await _patch_project(project_id, {"is_updating": True})

    workspace = EditWorkspace(uuid.UUID(project_id), repo_full_name, github_token)

    from app.build.build_logger import make_log_publisher
    log_fn = make_log_publisher(project_id, settings.REDIS_URL)
    log_fn("started", f"Applying update to {app_name}…")

    try:
        workspace.ensure()
        log_fn("info", "Workspace ready, running update agent…")

        from app.core.build_model import get_effective_build_model
        build_model = await get_effective_build_model(db, settings)

        # Build a condensed history of previous requests + what was done so the
        # agent understands what already exists and doesn't repeat or contradict it.
        contextual_prompt = prompt
        try:
            chat_doc = await db.collection("project_chats").document(project_id).get()
            if chat_doc.exists:
                all_msgs = (chat_doc.to_dict() or {}).get("messages", [])
                # Exclude the very last message (the current request — already in prompt)
                prior_msgs = all_msgs[-21:-1]
                lines = []
                for m in prior_msgs:
                    role = m.get("role", "user")
                    text = (m.get("text") or "")[:400].strip()
                    if not text or role == "error":
                        continue
                    if role == "user":
                        lines.append(f"User requested: {text}")
                    else:
                        lines.append(f"You replied / implemented: {text}")
                if lines:
                    contextual_prompt = (
                        "HISTORY OF PREVIOUS REQUESTS AND IMPLEMENTATIONS\n"
                        "(Use this to understand what already exists in the codebase "
                        "and avoid re-implementing or contradicting prior work)\n"
                        + "\n".join(lines)
                        + "\n\nCURRENT REQUEST\n"
                        + prompt
                    )
        except Exception:
            pass  # non-fatal — fall back to the bare prompt

        if build_model == "Qwen3":
            from app.build.build_agent import run_update_agent_ollama
            summary, tokens_used = run_update_agent_ollama(
                workspace=workspace.path,
                prompt=contextual_prompt,
                blueprint=blueprint_context,
                app_name=app_name,
                template_key=template_key,
                base_url=settings.OLLAMA_URL,
                model=settings.OLLAMA_MODEL,
                timeout=settings.OLLAMA_TIMEOUT,
                log_fn=log_fn,
            )
        elif build_model == "gemini":
            from app.build.build_agent import run_update_agent_gemini
            summary, tokens_used = run_update_agent_gemini(
                workspace=workspace.path,
                prompt=contextual_prompt,
                blueprint=blueprint_context,
                app_name=app_name,
                template_key=template_key,
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
                log_fn=log_fn,
            )
        elif build_model in ("gpt", "openai"):
            from app.build.build_agent import run_update_agent_openai
            summary, tokens_used = run_update_agent_openai(
                workspace=workspace.path,
                prompt=contextual_prompt,
                blueprint=blueprint_context,
                app_name=app_name,
                template_key=template_key,
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
                log_fn=log_fn,
            )
        else:
            summary, tokens_used = run_update_agent(
                workspace=workspace.path,
                prompt=contextual_prompt,
                blueprint=blueprint_context,
                app_name=app_name,
                template_key=template_key,
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.ANTHROPIC_MODEL,
                log_fn=log_fn,
            )
        logger.info("Update agent used %d tokens project=%s", tokens_used, project_id)
        await record_usage(db, user_id, tokens_used, is_update=True)

        log_fn("info", "Pushing changes to GitHub…")
        pushed = workspace.sync_to_github(
            commit_message=f"feat: {prompt[:60]}",  # original short prompt, not the full context
            push_url=push_url,
        )

        clean_summary = (summary or "").strip()
        if clean_summary.upper().startswith("DONE"):
            clean_summary = clean_summary[4:].lstrip(":").strip()

        now = datetime.now(timezone.utc)
        await _patch_project(project_id, {
            "is_updating": False,
            "build_error": None,
            "last_summary": clean_summary or None,
            "updated_at": now,
        })

        if pushed:
            log_fn("done", clean_summary or "Your app has been updated successfully!")
        else:
            agent_said = f'\n\nAgent response: "{clean_summary}"' if clean_summary else ""
            log_fn(
                "warning",
                f"The agent finished but made no file changes.{agent_said}\n\n"
                "Try breaking your request into a smaller, more specific step — "
                "e.g. 'Add a RiskScore badge to the task list item that shows High/Medium/Low.'"
            )
        logger.info("Update done project=%s pushed=%s prompt=%s", project_id, pushed, prompt[:40])
        return {"summary": summary, "pushed": pushed}

    except Exception as exc:
        from app.core.build_errors import sanitize_build_error
        build_err = sanitize_build_error(exc)
        logger.error("Update FAILED project=%s: %s", project_id, exc, exc_info=True)
        await _patch_project(project_id, {
            "is_updating": False,
            "build_error": build_err.message,
            "build_error_action": build_err.action,
            "updated_at": datetime.now(timezone.utc),
        })
        log_fn("error", f"Update failed: {build_err.message}")
        raise

    finally:
        workspace.cleanup()


@celery_app.task(name="app.workers.update_worker.apply_update", bind=True, max_retries=0)
def apply_update(self, project_id: str, prompt: str, user_id: str) -> dict:
    """Celery entry point — clone/pull → agent → commit+push."""
    logger.info("Update task started project=%s", project_id)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run(project_id, prompt, user_id))
    finally:
        loop.close()
