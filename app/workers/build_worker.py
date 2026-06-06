"""Build worker — Celery task that orchestrates the full build pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_TEMPLATE_KEYS = frozenset({"flutter", "react_native", "next"})


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("-").lower()
    return (slug or "forgefy-app")[:100]


def _deploy_cloudflare_pages(build_dir: Path, project_name: str) -> str | None:
    import os
    import subprocess

    settings = get_settings()
    if not settings.CLOUDFLARE_ACCOUNT_ID or not settings.CLOUDFLARE_API_TOKEN:
        return None

    env = os.environ.copy()
    env["CLOUDFLARE_ACCOUNT_ID"] = settings.CLOUDFLARE_ACCOUNT_ID
    env["CLOUDFLARE_API_TOKEN"] = settings.CLOUDFLARE_API_TOKEN
    env["CI"] = "true"

    try:
        result = subprocess.run(
            [
                "npx", "--yes", "wrangler", "pages", "deploy", str(build_dir),
                "--project-name", project_name,
                "--branch", "main",
                "--commit-dirty", "true",
            ],
            capture_output=True, text=True, env=env, timeout=120,
        )
        output = result.stdout + result.stderr
        match = re.search(r"https://[^\s]+\.pages\.dev", output)
        if match:
            url = match.group(0)
            logger.info("Cloudflare Pages deployed → %s", url)
            return url
        logger.warning("Wrangler ran but no pages.dev URL found in output")
        return None
    except Exception as exc:
        logger.warning("Cloudflare Pages deploy failed (non-fatal): %s", exc)
        return None


def _deploy_appetize(apk_path: Path, api_token: str) -> str | None:
    try:
        import httpx

        with apk_path.open("rb") as fh:
            resp = httpx.post(
                "https://api.appetize.io/v1/apps",
                auth=(api_token, ""),
                files={"file": (apk_path.name, fh, "application/octet-stream")},
                data={"platform": "android"},
                timeout=120,
            )
        if resp.status_code not in (200, 201):
            logger.warning("Appetize upload failed %s: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        url = data.get("appURL") or f"https://appetize.io/app/{data['publicKey']}"
        logger.info("Appetize preview → %s", url)
        return url
    except Exception as exc:
        logger.warning("Appetize upload failed (non-fatal): %s", exc)
        return None


def _deploy_expo_snack(workspace: Path, app_name: str) -> str | None:
    try:
        import json as _json
        import httpx

        _SKIP_DIRS = {"node_modules", "android", "ios", ".expo", ".git", ".next", "dist", "out"}
        _CODE_EXTS = {".js", ".jsx", ".ts", ".tsx"}
        _MAX_FILE_BYTES = 50_000
        _MAX_FILES = 60

        files: dict = {}
        for p in sorted(workspace.rglob("*")):
            if not p.is_file():
                continue
            rel_parts = p.relative_to(workspace).parts
            if set(rel_parts) & _SKIP_DIRS or rel_parts[0] in _SKIP_DIRS:
                continue
            if p.suffix not in _CODE_EXTS:
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > _MAX_FILE_BYTES:
                continue
            files["/".join(rel_parts)] = {"type": "CODE", "contents": content}
            if len(files) >= _MAX_FILES:
                break

        if not files:
            return None

        deps: dict = {}
        pkg_path = workspace / "package.json"
        if pkg_path.exists():
            deps = _json.loads(pkg_path.read_text()).get("dependencies", {})

        resp = httpx.post(
            "https://snack.expo.dev/api/v2/snack/save",
            json={"name": app_name, "description": "Built by Forgefy", "sdkVersion": "52.0.0", "files": files, "dependencies": deps},
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            return None
        snack_id = resp.json().get("id") or resp.json().get("hashId")
        if not snack_id:
            return None
        url = f"https://snack.expo.dev/{snack_id}"
        logger.info("Expo Snack preview → %s", url)
        return url
    except Exception as exc:
        logger.warning("Expo Snack deploy failed (non-fatal): %s", exc)
        return None


def _upload_artifact(artifact: Path, session_id: str) -> str | None:
    try:
        import io
        import zipfile
        import cloudinary
        import cloudinary.uploader

        settings = get_settings()
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True,
        )
        public_id = f"forgefy-builds/{session_id}/{artifact.stem}"
        if artifact.is_dir():
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in artifact.rglob("*"):
                    if p.is_file():
                        zf.write(p, p.relative_to(artifact))
            buf.seek(0)
            result = cloudinary.uploader.upload(buf, public_id=public_id, resource_type="raw", overwrite=True)
        else:
            result = cloudinary.uploader.upload(str(artifact), public_id=public_id, resource_type="raw", overwrite=True)
        url: str = result["secure_url"]
        logger.info("Artifact uploaded → %s", url)
        return url
    except Exception as exc:
        logger.warning("Cloudinary upload failed (non-fatal): %s", exc)
        return None


async def _load_approved_blueprint(session_id: str) -> dict:
    from app.db.firebase import get_firestore_client

    db = get_firestore_client()
    docs = (
        await db.collection("blueprints")
        .where("session_id", "==", session_id)
        .where("approved", "==", True)
        .get()
    )
    if not docs:
        raise ValueError(f"No approved blueprint for session {session_id}")
    latest = max(docs, key=lambda d: d.to_dict().get("created_at", ""))
    return latest.to_dict() | {"id": latest.id}


async def _patch_blueprint(blueprint_id: str, updates: dict) -> None:
    from app.db.firebase import get_firestore_client

    db = get_firestore_client()
    await db.collection("blueprints").document(blueprint_id).update(updates)




async def _run(session_id: str, project_id: str) -> dict:
    from app.build.workspace import Workspace
    from app.build.github_client import GitHubClient
    from app.build.build_agent import run_build_agent
    from app.build.build_logger import make_log_publisher
    from app.db.firebase import refresh_async_firestore_client
    from app.db.models.enums import SessionStatus
    from app.modules.voxa.state_machine import MeetingStateMachine

    settings = get_settings()
    db = refresh_async_firestore_client()  # bind fresh gRPC channel to this event loop

    # 1. Load approved blueprint
    bp = await _load_approved_blueprint(session_id)
    blueprint_id: str = bp["id"]
    json_output: dict = bp.get("json_output") or {}

    # 2. Determine template
    template_key = json_output.get("template", "next")
    if template_key not in _TEMPLATE_KEYS:
        template_key = "next"

    template_urls = {
        "flutter": settings.TEMPLATE_FLUTTER,
        "react_native": settings.TEMPLATE_REACT_NATIVE,
        "next": settings.TEMPLATE_NEXT,
    }
    template_url = template_urls[template_key]

    # 3. Derive app name — blueprint_generator always fills app_name now;
    #    fall back to first meaningful words of description, then a timestamp stub.
    raw = json_output.get("app_name") or ""
    if not raw:
        desc = (json_output.get("app_description") or "").strip()
        # Take the first five words of the description as a rough name
        words = desc.split()[:5]
        raw = " ".join(words) if words else f"app-{session_id[:8]}"
    app_name = _slugify(raw)

    # 4. Resolve GitHub token (validates personal token; falls back to system if invalid)
    from app.build.github_token import get_valid_github_token
    sess_doc = await db.collection("sessions").document(session_id).get()
    owner_id: str = sess_doc.to_dict()["user_id"] if sess_doc.exists else ""
    github_token = await get_valid_github_token(owner_id, settings.GITHUB_TOKEN)
    # True when user hasn't connected their GitHub — repo lands on the platform account
    using_platform_github: bool = github_token == settings.GITHUB_TOKEN

    # 5. Check the user's token quota before doing any expensive work
    from app.core.usage import get_user_tier, is_over_limit
    user_tier = await get_user_tier(db, owner_id)
    if await is_over_limit(db, owner_id, user_tier):
        from app.core.tiers import get_tier
        from app.build.build_logger import publish_user_event
        tier = get_tier(user_tier)
        error_msg = (
            f"You've used all {tier.monthly_tokens:,} tokens in your {tier.name} plan this month. "
            "Upgrade to continue building."
        )
        now = datetime.now(timezone.utc)
        await db.collection("projects").document(project_id).set({
            "owner_id": owner_id,
            "session_id": session_id,
            "blueprint_id": blueprint_id,
            "app_name": app_name,
            "template_key": template_key,
            "repo_full_name": "",
            "github_url": "",
            "repo_owner": "platform" if using_platform_github else "user",
            "preview_url": None,
            "artifact_url": None,
            "is_updating": False,
            "build_error": error_msg,
            "build_error_action": "support",
            "blueprint_context": json_output,
            "created_at": now,
            "updated_at": now,
        })
        publish_user_event(
            settings.REDIS_URL, owner_id, "quota_exceeded", error_msg,
            tier=user_tier, limit=tier.monthly_tokens,
        )
        logger.warning("Build blocked — token limit reached user=%s tier=%s", owner_id, user_tier)
        return {"project_id": project_id, "blocked": "token_limit"}

    # 6. Transition to BUILDING
    sm = MeetingStateMachine(db)
    await sm.transition(uuid.UUID(session_id), SessionStatus.BUILDING)
    await _patch_blueprint(blueprint_id, {"build_status": "IN_PROGRESS"})

    # 6. Create project doc early so frontend shows build in progress
    now = datetime.now(timezone.utc)
    await db.collection("projects").document(project_id).set({
        "owner_id": owner_id,
        "session_id": session_id,
        "blueprint_id": blueprint_id,
        "app_name": app_name,
        "template_key": template_key,
        "repo_full_name": "",
        "github_url": "",
        "repo_owner": "platform" if using_platform_github else "user",
        "preview_url": None,
        "artifact_url": None,
        "is_updating": True,
        "build_error": None,
        "blueprint_context": json_output,
        "created_at": now,
        "updated_at": now,
    })
    logger.info("Project stub created project=%s", project_id)

    log_fn = make_log_publisher(project_id, settings.REDIS_URL)
    log_fn("started", f"Starting build for {app_name} ({template_key})")

    # Log the feature plan so users immediately see what will be built
    features: list[str] = json_output.get("features") or []
    entities: list[str] = json_output.get("entities") or []
    if features:
        log_fn("info", f"Features to build: {', '.join(str(f) for f in features[:12])}")
    if entities:
        log_fn("info", f"Data models: {', '.join(str(e) for e in entities[:8])}")
    stack_label = {"flutter": "Flutter", "react_native": "React Native", "next": "Next.js"}.get(template_key, template_key)
    log_fn("info", f"Stack: {stack_label} · scaffolding project structure…")

    # Declared before try so the error handler can persist them even if a later step fails
    repo_url: str = ""
    repo_full_name: str = ""

    workspace = Workspace(uuid.UUID(session_id), template_key, template_url, git_token=github_token)
    try:
        # 7. Clone template
        workspace.clone()
        workspace.init_git()
        log_fn("info", "Template cloned — creating GitHub repository…")

        # 8. Create GitHub repo and push template code immediately so it's live on GitHub
        gh = GitHubClient(github_token)
        repo_data = gh.create_repo(
            name=app_name,
            description=(json_output.get("app_description") or "")[:100],
            private=True,
        )
        repo_url = repo_data["html_url"]
        repo_full_name = repo_data["full_name"]
        push_url = gh.get_push_url(repo_full_name)

        workspace.commit_all("chore: initial template")
        workspace.push(push_url)

        # Save repo info immediately — update prompts can now work even while the agent runs
        now = datetime.now(timezone.utc)
        await db.collection("projects").document(project_id).update({
            "repo_full_name": repo_full_name,
            "github_url": repo_url,
            "repo_owner": "platform" if using_platform_github else "user",
            "updated_at": now,
        })
        log_fn("info", f"Repository live on GitHub ({repo_url}). Running build agent…")

        # 9. Run Claude build agent — modifies files in the local workspace
        summary, tokens_used = run_build_agent(
            workspace=workspace.path,
            blueprint=json_output,
            app_name=app_name,
            template_key=template_key,
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.ANTHROPIC_MODEL,
            log_fn=log_fn,
        )
        logger.info("Build agent used %d tokens session=%s", tokens_used, session_id)

        # Record usage against the user's monthly quota
        from app.core.usage import record_usage
        await record_usage(db, owner_id, tokens_used, is_build=True)

        # 10. Push agent changes on top of the template commit
        log_fn("info", "Pushing agent changes to GitHub…")
        workspace.commit_all(f"feat: initial build by Forgefy\n\n{summary[:400]}")
        workspace.push(push_url)

        # 11. Preview + artifact (all non-fatal)
        artifact_url: str | None = None
        preview_url: str | None = None

        log_fn("info", "Building artifacts and deploying preview…")
        try:
            if template_key == "react_native":
                preview_url = _deploy_expo_snack(workspace.path, app_name)

            artifact_path = workspace.build_artifacts()

            if artifact_path:
                if template_key == "flutter" and artifact_path.is_file():
                    if settings.APPETIZE_API_TOKEN:
                        preview_url = _deploy_appetize(artifact_path, settings.APPETIZE_API_TOKEN)

                elif template_key == "next" and artifact_path.is_dir():
                    preview_url = _deploy_cloudflare_pages(artifact_path, app_name)

                elif template_key == "react_native" and artifact_path.is_dir():
                    cf_url = _deploy_cloudflare_pages(artifact_path, app_name)
                    if cf_url:
                        preview_url = cf_url

                if settings.CLOUDINARY_CLOUD_NAME:
                    artifact_url = _upload_artifact(artifact_path, session_id)

        except Exception as exc:
            logger.warning("Build/preview failed (non-fatal) session=%s: %s", session_id, exc)
            log_fn("warning", f"Preview deployment failed (code is still on GitHub): {exc}")

        # 12. Update blueprint
        bp_updates: dict = {
            "repo_url": repo_url,
            "repo_name": app_name,
            "build_summary": summary,
            "build_status": "SUCCESS",
        }
        if artifact_url:
            bp_updates["artifact_url"] = artifact_url
        if preview_url:
            bp_updates["preview_url"] = preview_url
        await _patch_blueprint(blueprint_id, bp_updates)

        # 13. Update project doc with final values
        now = datetime.now(timezone.utc)
        await db.collection("projects").document(project_id).update({
            "repo_full_name": repo_full_name,
            "github_url": repo_url,
            "repo_owner": "platform" if using_platform_github else "user",
            "preview_url": preview_url,
            "artifact_url": artifact_url,
            "is_updating": False,
            "build_error": None,
            "updated_at": now,
        })

        log_fn("done", "Build complete! Code is live on GitHub.")
        logger.info("Build SUCCESS session=%s repo=%s preview=%s", session_id, repo_url, preview_url)

        return {
            "project_id": project_id,
            "repo_url": repo_url,
            "preview_url": preview_url,
            "artifact_url": artifact_url,
            "build_summary": summary,
        }

    except Exception as exc:
        from app.core.build_errors import sanitize_build_error
        build_err = sanitize_build_error(exc)
        logger.error("Build FAILED session=%s: %s", session_id, exc, exc_info=True)

        await _patch_blueprint(blueprint_id, {
            "build_status": "FAILED",
            "build_error": build_err.message,
        })

        now = datetime.now(timezone.utc)
        proj_err: dict = {
            "is_updating": False,
            "build_error": build_err.message,
            "build_error_action": build_err.action,
            "updated_at": now,
        }
        # If the GitHub repo was created before the failure, save the link so it isn't lost
        if repo_url:
            proj_err["github_url"] = repo_url
            proj_err["repo_full_name"] = repo_full_name
            proj_err["repo_owner"] = "platform" if using_platform_github else "user"
        await db.collection("projects").document(project_id).update(proj_err)

        log_fn("error", f"Build failed: {build_err.message}")
        raise

    finally:
        workspace.cleanup()


@celery_app.task(name="app.workers.build_worker.run_build", bind=True, max_retries=0)
def run_build(self, session_id: str, project_id: str) -> dict:
    """Celery entry point — clone → agent → push → build → upload."""
    logger.info("Build task started session=%s project=%s", session_id, project_id)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run(session_id, project_id))
    finally:
        loop.close()
