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

        summary, tokens_used = run_update_agent(
            workspace=workspace.path,
            prompt=prompt,
            blueprint=blueprint_context,
            app_name=app_name,
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.ANTHROPIC_MODEL,
            log_fn=log_fn,
        )
        logger.info("Update agent used %d tokens project=%s", tokens_used, project_id)
        await record_usage(db, user_id, tokens_used, is_update=True)

        log_fn("info", "Pushing changes to GitHub…")
        workspace.sync_to_github(
            commit_message=f"feat: {prompt[:60]}",
            push_url=push_url,
        )

        now = datetime.now(timezone.utc)
        await _patch_project(project_id, {
            "is_updating": False,
            "build_error": None,
            "updated_at": now,
        })

        log_fn("done", "Update complete! Changes pushed to GitHub.")
        logger.info("Update done project=%s prompt=%s", project_id, prompt[:40])
        return {"summary": summary}

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
