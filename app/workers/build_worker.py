"""Build worker — Celery task that orchestrates the full build pipeline."""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_TEMPLATE_KEYS = frozenset({"flutter", "react_native", "next"})


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("-").lower()
    return (slug or "forgefy-app")[:100]


def _deploy_cloudflare_pages(build_dir: Path, project_name: str) -> str | None:
    """Deploy a static directory to Cloudflare Pages via wrangler; return preview URL."""
    import os
    import subprocess

    settings = get_settings()
    if not settings.CLOUDFLARE_ACCOUNT_ID or not settings.CLOUDFLARE_API_TOKEN:
        return None

    env = os.environ.copy()
    env["CLOUDFLARE_ACCOUNT_ID"] = settings.CLOUDFLARE_ACCOUNT_ID
    env["CLOUDFLARE_API_TOKEN"] = settings.CLOUDFLARE_API_TOKEN
    env["CI"] = "true"  # prevents interactive prompts

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
        logger.debug("Wrangler output: %s", output[:500])

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
    """Upload an APK to Appetize.io; return the browser-playable app URL."""
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
    """Publish source files to Expo Snack; return the snack.expo.dev preview URL."""
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
            logger.warning("Expo Snack: no source files found in workspace")
            return None

        # Pull dependencies straight from package.json
        deps: dict = {}
        pkg_path = workspace / "package.json"
        if pkg_path.exists():
            deps = _json.loads(pkg_path.read_text()).get("dependencies", {})

        payload = {
            "name": app_name,
            "description": "Built by Forgefy",
            "sdkVersion": "52.0.0",
            "files": files,
            "dependencies": deps,
        }

        resp = httpx.post(
            "https://snack.expo.dev/api/v2/snack/save",
            json=payload,
            timeout=30,
        )

        if resp.status_code not in (200, 201):
            logger.warning("Expo Snack save failed %s: %s", resp.status_code, resp.text[:200])
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
    """Upload a file or directory (zipped) to Cloudinary; return the secure URL."""
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
            result = cloudinary.uploader.upload(
                buf,
                public_id=public_id,
                resource_type="raw",
                overwrite=True,
            )
        else:
            result = cloudinary.uploader.upload(
                str(artifact),
                public_id=public_id,
                resource_type="raw",
                overwrite=True,
            )

        url: str = result["secure_url"]
        logger.info("Artifact uploaded to Cloudinary → %s", url)
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


async def _get_user_github_token(user_id: str) -> str | None:
    from app.db.firebase import get_firestore_client

    db = get_firestore_client()
    doc = await db.collection("users").document(user_id).get()
    if doc.exists:
        return doc.to_dict().get("github_access_token")
    return None


async def _run(session_id: str) -> dict:
    from app.build.workspace import Workspace
    from app.build.github_client import GitHubClient
    from app.build.build_agent import run_build_agent
    from app.db.firebase import get_firestore_client
    from app.db.models.enums import SessionStatus
    from app.modules.voxa.state_machine import MeetingStateMachine

    settings = get_settings()

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

    # 3. Derive app name / repo name
    raw = (
        json_output.get("app_name")
        or (json_output.get("app_description") or "")[:40]
        or f"app-{session_id[:8]}"
    )
    app_name = _slugify(raw)

    # 3b. Resolve GitHub token (user's personal if linked, else system)
    sess_doc = await get_firestore_client().collection("sessions").document(session_id).get()
    owner_id: str = sess_doc.to_dict()["user_id"] if sess_doc.exists else ""
    github_token = (await _get_user_github_token(owner_id)) or settings.GITHUB_TOKEN

    # 4. Transition to BUILDING
    db = get_firestore_client()
    sm = MeetingStateMachine(db)
    await sm.transition(uuid.UUID(session_id), SessionStatus.BUILDING)
    await _patch_blueprint(blueprint_id, {"build_status": "IN_PROGRESS"})

    # 5. Clone template
    workspace = Workspace(uuid.UUID(session_id), template_key, template_url)
    workspace.clone()

    try:
        workspace.init_git()

        # 6. Run Claude build agent
        summary = run_build_agent(
            workspace=workspace.path,
            blueprint=json_output,
            app_name=app_name,
            template_key=template_key,
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.ANTHROPIC_MODEL,
        )

        # 7. Create GitHub repo and push (use user's personal token if linked)
        gh = GitHubClient(github_token)
        repo_data = gh.create_repo(
            name=app_name,
            description=(json_output.get("app_description") or "")[:100],
            private=True,
        )
        repo_url: str = repo_data["html_url"]
        push_url = gh.get_push_url(repo_data["full_name"])

        workspace.commit_all(f"feat: initial build by Forgefy\n\n{summary[:400]}")
        workspace.push(push_url)

        # 8. Preview + artifact (all non-fatal — code is already in GitHub)
        artifact_url: str | None = None
        preview_url: str | None = None

        try:
            # React Native: Expo Snack from source immediately (no build required)
            if template_key == "react_native":
                preview_url = _deploy_expo_snack(workspace.path, app_name)

            artifact_path = workspace.build_artifacts()

            if artifact_path:
                if template_key == "flutter" and artifact_path.is_file():
                    # Flutter APK → Appetize browser simulator
                    if settings.APPETIZE_API_TOKEN:
                        preview_url = _deploy_appetize(artifact_path, settings.APPETIZE_API_TOKEN)

                elif template_key == "next" and artifact_path.is_dir():
                    # Next.js static export → Cloudflare Pages live URL
                    preview_url = _deploy_cloudflare_pages(artifact_path, app_name)

                elif template_key == "react_native" and artifact_path.is_dir():
                    # Expo web export → Cloudflare Pages (better than Snack if available)
                    cf_url = _deploy_cloudflare_pages(artifact_path, app_name)
                    if cf_url:
                        preview_url = cf_url

                # Upload raw artifact (APK / zip) to Cloudinary
                if settings.CLOUDINARY_CLOUD_NAME:
                    artifact_url = _upload_artifact(artifact_path, session_id)

        except Exception as exc:
            logger.warning("Build/preview failed (non-fatal) session=%s: %s", session_id, exc)

        # 9. Persist results
        updates: dict = {
            "repo_url": repo_url,
            "repo_name": app_name,
            "build_summary": summary,
            "build_status": "SUCCESS",
        }
        if artifact_url:
            updates["artifact_url"] = artifact_url
        if preview_url:
            updates["preview_url"] = preview_url

        await _patch_blueprint(blueprint_id, updates)

        # 10. Save project document to Firestore
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        project_id = str(uuid.uuid4())
        project_doc: dict = {
            "owner_id": owner_id,
            "session_id": session_id,
            "blueprint_id": blueprint_id,
            "app_name": app_name,
            "template_key": template_key,
            "repo_full_name": repo_data["full_name"],
            "github_url": repo_url,
            "preview_url": preview_url,
            "artifact_url": artifact_url,
            "is_updating": False,
            "blueprint_context": json_output,
            "created_at": now,
            "updated_at": now,
        }
        await db.collection("projects").document(project_id).set(project_doc)
        logger.info("Project saved project=%s", project_id)

        logger.info(
            "Build SUCCESS session=%s repo=%s preview=%s",
            session_id, repo_url, preview_url,
        )
        return {
            "project_id": project_id,
            "repo_url": repo_url,
            "preview_url": preview_url,
            "artifact_url": artifact_url,
            "build_summary": summary,
        }

    except Exception as exc:
        logger.error("Build FAILED session=%s: %s", session_id, exc, exc_info=True)
        await _patch_blueprint(blueprint_id, {
            "build_status": "FAILED",
            "build_error": str(exc)[:500],
        })
        raise

    finally:
        # Keep workspace on disk for potential future edits
        pass


@celery_app.task(name="app.workers.build_worker.run_build", bind=True, max_retries=0)
def run_build(self, session_id: str) -> dict:
    """Celery entry point — clone → agent → push → build → upload."""
    logger.info("Build task started session=%s", session_id)
    return asyncio.run(_run(session_id))
