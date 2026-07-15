"""Update worker — applies a user prompt to an existing project."""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

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
    from app.build.build_agent import run_update_agent
    from app.build.cancel import clear_cancel, is_cancelled
    from app.build.workspace import EditWorkspace
    from app.core.usage import record_usage
    from app.db.firebase import refresh_async_firestore_client

    settings = get_settings()
    # Wipe any stale cancel flag left over from a previous crashed/timed-out run
    # before we start fresh work. Any legitimate cancel from the new run will
    # be set after this point, so this is safe.
    clear_cancel(settings.REDIS_URL, project_id)

    def cancel_fn() -> bool:
        return is_cancelled(settings.REDIS_URL, project_id)

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

    # Check token quota before doing any expensive work. Free users over budget
    # are blocked; paid users are downgraded to the free model and keep going
    # (see app/core/usage.py evaluate_quota). quota_forced_model, when set,
    # overrides the resolved build_model below.
    from app.build.build_logger import publish_user_event
    from app.core.tiers import get_tier
    from app.core.usage import evaluate_quota
    quota = await evaluate_quota(db, settings, user_id)
    tier = get_tier(quota.tier_key)
    quota_forced_model: str | None = None
    if quota.action == "block":
        await _patch_project(project_id, {
            "build_error": quota.message,
            "build_error_action": "support",
            "updated_at": datetime.now(UTC),
        })
        publish_user_event(
            settings.REDIS_URL, user_id, "quota_exceeded", quota.message,
            tier=quota.tier_key, limit=tier.monthly_tokens,
        )
        logger.warning("Update blocked — token limit reached user=%s tier=%s", user_id, quota.tier_key)
        return {"blocked": "token_limit"}
    if quota.action == "downgrade":
        quota_forced_model = quota.forced_model
        publish_user_event(
            settings.REDIS_URL, user_id, "quota_downgrade", quota.message,
            tier=quota.tier_key, limit=tier.monthly_tokens, forced_model=quota.forced_model,
        )
        logger.info(
            "Update over quota — downgraded to free model=%s user=%s tier=%s",
            quota.forced_model, user_id, quota.tier_key,
        )

    await _patch_project(project_id, {"is_updating": True})

    workspace = EditWorkspace(uuid.UUID(project_id), repo_full_name, github_token)

    from app.build.build_logger import make_log_publisher
    log_fn = make_log_publisher(project_id, settings.REDIS_URL)
    if quota_forced_model:
        log_fn("warning", quota.message)
    log_fn("started", f"Applying update to {app_name}…")

    try:
        workspace.ensure()
        log_fn("info", "Workspace ready, running update agent…")

        from app.core.build_model import get_effective_build_model
        build_model = await get_effective_build_model(db, settings, user_id=user_id)
        # A paid user over quota is downgraded to the free model for this update.
        if quota_forced_model:
            build_model = quota_forced_model

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
            from app.ai.qwen import using_openrouter

            if using_openrouter():
                from app.build.build_agent import run_update_agent_openrouter
                summary, tokens_used = run_update_agent_openrouter(
                    workspace=workspace.path,
                    prompt=contextual_prompt,
                    blueprint=blueprint_context,
                    app_name=app_name,
                    template_key=template_key,
                    log_fn=log_fn,
                    cancel_fn=cancel_fn,
                )
            else:
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
                    cancel_fn=cancel_fn,
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
                cancel_fn=cancel_fn,
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
                cancel_fn=cancel_fn,
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
                cancel_fn=cancel_fn,
            )
        logger.info("Update agent used %d tokens project=%s", tokens_used, project_id)
        await record_usage(db, user_id, tokens_used, is_update=True)

        # If a database is connected for this project, keep .env in sync with
        # the real values (see app/build/workspace.py) in case the agent
        # rewrote them.
        if project.get("supabase_url") and project.get("supabase_anon_key"):
            workspace.write_supabase_env(
                template_key, project["supabase_url"], project["supabase_anon_key"]
            )
        if project.get("neon_data_api_url"):
            workspace.write_neon_env(template_key, project["neon_data_api_url"])
        if project.get("firebase_project_id"):
            workspace.write_firebase_env(template_key, {
                "apiKey": project.get("firebase_api_key"),
                "authDomain": project.get("firebase_auth_domain"),
                "projectId": project.get("firebase_project_id"),
                "storageBucket": project.get("firebase_storage_bucket"),
                "messagingSenderId": project.get("firebase_messaging_sender_id"),
                "appId": project.get("firebase_app_id"),
            })

        was_stopped = cancel_fn()
        clear_cancel(settings.REDIS_URL, project_id)

        pushed = workspace.sync_to_github(
            commit_message=f"feat: {prompt[:60]}",  # original short prompt, not the full context
            push_url=push_url,
        )
        if pushed:
            log_fn("info", "Changes pushed to GitHub.")

        raw = (summary or "").strip()
        # Extract the human-readable part of the summary.
        # The validator output is "DESIGN AUDIT: ... VALIDATED: <message>" —
        # find the VALIDATED section wherever it appears, not just at the start.
        upper = raw.upper()
        val_idx = upper.find("VALIDATED:")
        if val_idx != -1:
            clean_summary = raw[val_idx + len("VALIDATED:"):].lstrip().strip()
        elif upper.startswith("DONE"):
            clean_summary = raw[4:].lstrip(":").strip()
        else:
            clean_summary = raw

        hit_limit = "Agent reached iteration limit." in (summary or "")

        # Capture .env.local placeholder content before workspace is deleted so the
        # code viewer can display it (gitignored, won't be on GitHub).
        env_local_path = workspace.path / ".env.local"
        env_local_content = env_local_path.read_text(encoding="utf-8") if env_local_path.exists() else None

        now = datetime.now(UTC)
        patch: dict = {
            "is_updating": False,
            "build_error": None,
            "last_summary": clean_summary or None,
            "updated_at": now,
        }
        if env_local_content is not None:
            patch["env_local_template"] = env_local_content
        await _patch_project(project_id, patch)

        if was_stopped:
            if pushed:
                log_fn("warning", "Agent stopped — partial changes have been saved to GitHub.")
            else:
                log_fn("warning", "Agent stopped before any changes were made.")
            logger.info("Update stopped by user project=%s pushed=%s", project_id, pushed)
        elif pushed and not hit_limit:
            log_fn("done", clean_summary or "Your app has been updated successfully!")
            # Automatically rebuild the preview so users don't have to click manually
            try:
                from app.workers.build_worker import build_preview
                build_preview.apply_async(
                    args=[project_id, user_id],
                    queue="build",
                    countdown=3,  # small delay so the push is visible on GitHub before we clone
                )
                log_fn("info", "Preview build queued…")
                logger.info("Auto-queued preview build project=%s", project_id)
            except Exception as preview_exc:
                logger.warning("Could not queue preview build project=%s: %s", project_id, preview_exc)
        elif pushed and hit_limit:
            log_fn(
                "warning",
                "The agent pushed partial changes but hit its step limit before finishing. "
                "Send the same request again to continue — it will pick up where it left off."
            )
            logger.warning("Preview build skipped — agent hit iteration limit project=%s", project_id)
        else:
            # Distinguish between "agent confirmed everything already correct" (not an error)
            # vs "agent genuinely failed to act on the request".
            summary_upper = upper  # already computed above
            already_done = (
                "all checks passed" in summary_upper
                or "no issues found" in summary_upper
                or "no changes" in summary_upper
                or ("validated" in summary_upper and "no other issues" in summary_upper)
            )
            if already_done:
                log_fn("done", clean_summary or "Everything is already up to date — no changes needed.")
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
            "updated_at": datetime.now(UTC),
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
