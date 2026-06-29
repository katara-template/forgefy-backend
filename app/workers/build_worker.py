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


class _BuildFixExhausted(RuntimeError):
    """Raised when the auto-fix loop gives up — lets outer handlers show the right message."""


# ---------------------------------------------------------------------------
# Auto-fix helpers
# ---------------------------------------------------------------------------

def _extract_dart_errors(error_msg: str) -> str:
    """Pull out just the Dart compiler error lines from a Flutter/Gradle build log."""
    import re as _re
    # Dart errors look like: lib/foo/bar.dart:12:5: Error: ...
    dart_lines = _re.findall(r"(?:lib|test)/[^\n]+\.dart:\d+:\d+:[^\n]+", error_msg)
    if dart_lines:
        return "\n".join(dict.fromkeys(dart_lines))  # deduplicate, preserve order
    # fallback: last 3000 chars of combined output (stdout+stderr both captured now)
    return error_msg[-3000:].strip()


def _make_fix_prompt(error_msg: str, template_key: str) -> str:
    """Turn a raw compilation error into a precise agent fix prompt."""
    framework = {
        "flutter": "Flutter/Dart",
        "next": "Next.js/TypeScript",
        "react_native": "React Native/TypeScript",
    }.get(template_key, template_key)

    if template_key == "flutter":
        # Extract the precise Dart error lines so the agent knows exactly which files to fix
        dart_errors = _extract_dart_errors(error_msg)
        snippet = dart_errors + "\n\n--- full tail ---\n" + error_msg[-1500:].strip()
    else:
        snippet = error_msg[-3000:].strip()

    hints = {
        "flutter": (
            "• BorderRadius.circular(), NOT BorderRadiusGeometry.circular()\n"
            "• Colors.blue, NOT Color.blue\n"
            "• Add missing 'const' where required by the compiler\n"
            "• Fix null-safety issues (add ?, ! or explicit null checks)\n"
            "• Verify package names in pubspec.yaml match the imports\n"
            "• Undefined class/method → check import statements and pubspec.yaml\n"
            "• 'Target kernel_snapshot_program failed' → Dart type/syntax error in the listed file"
        ),
        "next": (
            "• Add 'use client' for components that use hooks or browser APIs\n"
            "• Fix @/ import aliases (must match tsconfig paths)\n"
            "• Resolve TypeScript type mismatches and missing generics\n"
            "• Missing packages → add to package.json dependencies"
        ),
        "react_native": (
            "• Fix incorrect import paths\n"
            "• Resolve TypeScript type errors\n"
            "• Add missing package imports"
        ),
    }.get(template_key, "• Fix all compilation errors listed above")

    return f"""The {framework} app failed to compile. Diagnose and fix ALL errors so the build succeeds.
COMPILATION ERROR OUTPUT:
{snippet}

INSTRUCTIONS:
1. Call list_files('.') to understand the project structure.
2. Read EVERY file mentioned in the error output above (the exact file:line references).
3. Fix the exact errors — wrong API names, missing imports, type mismatches, undefined identifiers.
4. Only modify existing files. Do NOT add new packages or change pubspec.yaml unless a package is clearly missing.
5. After fixing everything output: DONE: <brief summary of fixes>

COMMON {framework.upper()} MISTAKES TO CHECK:
{hints}"""


def _make_fix_prompt_with_hint(error_msg: str, template_key: str, retry_hint: bool = False) -> str:
    """Like _make_fix_prompt but prepends a stronger warning when the same error repeats."""
    base = _make_fix_prompt(error_msg, template_key)
    if not retry_hint:
        return base
    warning = (
        "⚠️  PREVIOUS FIX DID NOT WORK — the EXACT SAME error is still occurring after your last attempt.\n"
        "You MUST try a completely different approach:\n"
        "  • Read MORE surrounding files to find the real root cause\n"
        "  • Check that the file you edited actually exists at the path you used\n"
        "  • Look for the error upstream — it may originate in a file you haven't read yet\n"
        "  • Verify all imports resolve to real files in the workspace\n\n"
    )
    return warning + base


def _run_agent_fix(
    workspace_path: "Path",
    error_msg: str,
    template_key: str,
    app_name: str,
    blueprint: dict,
    settings,
    log_fn,
    retry_hint: bool = False,
) -> None:
    """Run the update agent to fix compilation errors (synchronous, safe for Celery)."""
    fix_prompt = _make_fix_prompt_with_hint(error_msg, template_key, retry_hint)

    if settings.BUILD_MODEL == "Qwen3":
        from app.build.build_agent import run_update_agent_ollama
        run_update_agent_ollama(
            workspace=workspace_path,
            prompt=fix_prompt,
            blueprint=blueprint,
            app_name=app_name,
            template_key=template_key,
            base_url=settings.OLLAMA_URL,
            model=settings.OLLAMA_MODEL,
            timeout=settings.OLLAMA_TIMEOUT,
            log_fn=log_fn,
        )
    elif settings.BUILD_MODEL == "gemini":
        from app.build.build_agent import run_update_agent_gemini
        run_update_agent_gemini(
            workspace=workspace_path,
            prompt=fix_prompt,
            blueprint=blueprint,
            app_name=app_name,
            template_key=template_key,
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            log_fn=log_fn,
        )
    elif settings.BUILD_MODEL in ("gpt", "openai"):
        from app.build.build_agent import run_update_agent_openai
        run_update_agent_openai(
            workspace=workspace_path,
            prompt=fix_prompt,
            blueprint=blueprint,
            app_name=app_name,
            template_key=template_key,
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            log_fn=log_fn,
        )
    else:
        from app.build.build_agent import run_update_agent
        run_update_agent(
            workspace=workspace_path,
            prompt=fix_prompt,
            blueprint=blueprint,
            app_name=app_name,
            template_key=template_key,
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.ANTHROPIC_MODEL,
            log_fn=log_fn,
        )


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("-").lower()
    return (slug or "forgefy-app")[:100]


def _cf_project_name(name: str) -> str:
    """Cloudflare Pages project names: lowercase, alphanumeric + hyphens, max 28 chars."""
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return (slug or "forgefy-app")[:28].rstrip("-")


def _deploy_cloudflare_pages(build_dir: Path, project_name: str) -> str | None:
    import os
    import subprocess

    settings = get_settings()
    if not settings.CLOUDFLARE_ACCOUNT_ID or not settings.CLOUDFLARE_API_TOKEN:
        logger.warning("Cloudflare Pages skipped — CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN not set")
        return None

    cf_name = _cf_project_name(project_name)
    env = os.environ.copy()
    env["CLOUDFLARE_ACCOUNT_ID"] = settings.CLOUDFLARE_ACCOUNT_ID
    env["CLOUDFLARE_API_TOKEN"] = settings.CLOUDFLARE_API_TOKEN
    env["CI"] = "true"
    # Stop wrangler from trying to open a browser for auth
    env["WRANGLER_SEND_METRICS"] = "false"

    try:
        result = subprocess.run(
            [
                "npx", "--yes", "wrangler@3", "pages", "deploy", str(build_dir),
                "--project-name", cf_name,
                "--branch", "main",
                "--commit-dirty", "true",
            ],
            capture_output=True, text=True, env=env, timeout=180,
        )
        output = result.stdout + result.stderr
        logger.info("Wrangler output (rc=%d):\n%s", result.returncode, output[-1500:])

        match = re.search(r"https://[^\s]+\.pages\.dev[^\s]*", output)
        if match:
            url = match.group(0).rstrip(".")
            logger.info("Cloudflare Pages deployed → %s", url)
            return url

        if result.returncode != 0:
            print(f"Wrangler deploy failed with code {result.returncode}:\n{output}")
            logger.warning("Wrangler exited with code %d — deploy failed", result.returncode)
        else:
            print(f"Wrangler deploy succeeded but no pages.dev URL found:\n{output}")
            logger.warning("Wrangler succeeded but no pages.dev URL found in output")
        return None
    except subprocess.TimeoutExpired as exc:
        # print exception error message and output for debugging
        print(f"Wrangler deploy timed out: {exc}")
        logger.warning("Cloudflare Pages deploy timed out after 180 s")
        return None
    except Exception as exc:
        print(f"Wrangler deploy failed: {exc}")
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
        log_fn("info", f"Repository live on GitHub ({repo_url}). Generating design system…")

        # 8.5 Phase 0.5 — materialise design system files before the build agent reads them
        try:
            from app.build.design_system import generate_design_system_files
            generate_design_system_files(
                workspace_path=workspace.path,
                blueprint=json_output,
                framework=template_key,
                log_fn=log_fn,
            )
        except Exception as _ds_exc:
            logger.warning("Design system generation failed (non-fatal): %s", _ds_exc)
            log_fn("info", "Design system generation skipped — build agent will create styles.")

        log_fn("info", "Running build agent…")

        # 9. Run build agent — model selected by BUILD_MODEL (independent of BP_MODEL)
        if settings.BUILD_MODEL == "Qwen3":
            from app.build.build_agent import run_build_agent_ollama
            summary, tokens_used = run_build_agent_ollama(
                workspace=workspace.path,
                blueprint=json_output,
                app_name=app_name,
                template_key=template_key,
                base_url=settings.OLLAMA_URL,
                model=settings.OLLAMA_MODEL,
                timeout=settings.OLLAMA_TIMEOUT,
                log_fn=log_fn,
            )
        elif settings.BUILD_MODEL == "gemini":
            from app.build.build_agent import run_build_agent_gemini
            summary, tokens_used = run_build_agent_gemini(
                workspace=workspace.path,
                blueprint=json_output,
                app_name=app_name,
                template_key=template_key,
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
                log_fn=log_fn,
            )
        elif settings.BUILD_MODEL in ("gpt", "openai"):
            from app.build.build_agent import run_build_agent_openai
            summary, tokens_used = run_build_agent_openai(
                workspace=workspace.path,
                blueprint=json_output,
                app_name=app_name,
                template_key=template_key,
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
                log_fn=log_fn,
            )
        else:
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

            log_fn("info", f"Compiling {stack_label} project (this may take a few minutes)…")
            artifact_path = None
            _fix_attempt = 0
            _prev_err: str | None = None
            _same_err_streak = 0
            _MAX_FIX = 10
            _MAX_SAME_STREAK = 3  # only give up if identical error 3 times in a row
            from app.build.cancel import is_cancelled, clear_cancel
            while True:
                if is_cancelled(settings.REDIS_URL, session_id):
                    clear_cancel(settings.REDIS_URL, session_id)
                    log_fn("warning", "Build stopped by user.")
                    return {}
                try:
                    artifact_path = workspace.build_artifacts()
                    break  # compiled clean
                except Exception as _compile_exc:
                    _err = str(_compile_exc)
                    _fix_attempt += 1
                    if _fix_attempt > _MAX_FIX:
                        log_fn("warning", f"Build could not be fixed after {_MAX_FIX} attempts. Try sending a chat message describing what to fix.")
                        raise _BuildFixExhausted(f"Auto-fix exhausted after {_MAX_FIX} attempts") from _compile_exc
                    if _err == _prev_err:
                        _same_err_streak += 1
                    else:
                        _same_err_streak = 0
                    _prev_err = _err
                    if _same_err_streak >= _MAX_SAME_STREAK:
                        log_fn("warning", "The same error persisted after several fix attempts. Try sending a chat message to fix it manually.")
                        raise _BuildFixExhausted(f"Auto-fix stalled — same error after {_same_err_streak} consecutive attempts") from _compile_exc
                    retry_context = " (same error — trying a different approach)" if _same_err_streak > 0 else ""
                    log_fn("info", f"Compilation failed — running auto-fix (attempt {_fix_attempt}){retry_context}…")
                    logger.warning("Auto-fix triggered session=%s attempt=%d streak=%d: %s", session_id, _fix_attempt, _same_err_streak, _err[:200])
                    _run_agent_fix(workspace.path, _err, template_key, app_name, json_output, settings, log_fn, retry_hint=_same_err_streak > 0)
                    workspace.commit_all("fix: auto-fix compilation errors")
                    workspace.push(push_url)
                    log_fn("info", "Fixes applied — retrying compilation…")

            if artifact_path:
                log_fn("info", "Compilation successful — deploying preview…")
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
            else:
                log_fn("warning", "Compilation produced no output — preview unavailable. Check the GitHub repo for errors.")

        except Exception as exc:
            error_detail = str(exc)
            # Show the last 800 chars of the error — compilation errors are at the end
            snippet = error_detail[-800:].strip() if len(error_detail) > 800 else error_detail.strip()
            logger.warning("Build/preview failed (non-fatal) session=%s: %s", session_id, exc)
            log_fn("warning", f"Preview build failed (code is still on GitHub):\n{snippet}")

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

        clean_summary = (summary or "").strip()
        if clean_summary.upper().startswith("DONE"):
            clean_summary = clean_summary[4:].lstrip(":").strip()
        log_fn("done", clean_summary or "Your app has been built successfully!")
        logger.info("Build SUCCESS session=%s repo=%s preview=%s", session_id, repo_url, preview_url)

        return {
            "project_id": project_id,
            "repo_url": repo_url,
            "preview_url": preview_url,
            "artifact_url": artifact_url,
            "build_summary": summary,
        }

    except _BuildFixExhausted as exc:
        # Auto-fix ran out of attempts — code is on GitHub but wouldn't compile.
        # Do NOT say "the agent will attempt to fix" — it already tried and stopped.
        user_msg = (
            "The preview couldn't be compiled automatically. "
            "Your code is saved on GitHub — send a chat message like "
            "'Fix the build errors' and the agent will try again."
        )
        logger.error("Build auto-fix exhausted session=%s: %s", session_id, exc)
        await _patch_blueprint(blueprint_id, {"build_status": "FAILED", "build_error": user_msg})
        now = datetime.now(timezone.utc)
        proj_err: dict = {
            "is_updating": False,
            "build_error": user_msg,
            "build_error_action": "user_fix",
            "updated_at": now,
        }
        if repo_url:
            proj_err["github_url"] = repo_url
            proj_err["repo_full_name"] = repo_full_name
            proj_err["repo_owner"] = "platform" if using_platform_github else "user"
        await db.collection("projects").document(project_id).update(proj_err)
        log_fn("warning", user_msg)
        return {}

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


async def _run_preview(project_id: str, user_id: str) -> dict:
    """Clone the project repo, compile, deploy preview, update Firestore."""
    from app.build.workspace import EditWorkspace
    from app.build.build_logger import make_log_publisher
    from app.build.github_token import get_valid_github_token
    from app.db.firebase import refresh_async_firestore_client

    settings = get_settings()
    db = refresh_async_firestore_client()

    doc = await db.collection("projects").document(project_id).get()
    if not doc.exists:
        raise ValueError(f"Project {project_id} not found")
    project_data = doc.to_dict()

    repo_full_name: str = project_data["repo_full_name"]
    template_key: str = project_data.get("template_key", "next")
    app_name: str = project_data["app_name"]
    session_id: str = project_data.get("session_id", project_id)

    github_token = await get_valid_github_token(user_id, settings.GITHUB_TOKEN)
    log_fn = make_log_publisher(project_id, settings.REDIS_URL)
    stack_label = {"flutter": "Flutter", "react_native": "React Native", "next": "Next.js"}.get(template_key, template_key)

    # Mark project as busy so the frontend log panel activates and the WebSocket
    # delivers updates. Cleared in both the success and failure paths below.
    await db.collection("projects").document(project_id).update({
        "is_updating": True,
        "build_error": None,
        "updated_at": datetime.now(timezone.utc),
    })
    log_fn("started", f"Building preview for {app_name}…")

    workspace = EditWorkspace(uuid.UUID(project_id), repo_full_name, github_token)
    push_url = f"https://{github_token}@github.com/{repo_full_name}.git"
    try:
        workspace.ensure()
        log_fn("info", f"Compiling {stack_label} project (this may take a few minutes)…")

        artifact_path = None
        blueprint = project_data.get("blueprint_context") or {}
        _fix_attempt = 0
        _prev_err: str | None = None
        _same_err_streak = 0
        _MAX_FIX = 10
        _MAX_SAME_STREAK = 3
        from app.build.cancel import is_cancelled, clear_cancel
        while True:
            if is_cancelled(settings.REDIS_URL, project_id):
                clear_cancel(settings.REDIS_URL, project_id)
                log_fn("warning", "Build stopped by user.")
                return {}
            try:
                artifact_path = workspace.build_artifacts(template_key)
                break  # compiled clean
            except Exception as _compile_exc:
                _err = str(_compile_exc)
                _fix_attempt += 1
                if _fix_attempt > _MAX_FIX:
                    log_fn("warning", f"Build could not be fixed after {_MAX_FIX} attempts. Try sending a chat message describing what to fix.")
                    raise _BuildFixExhausted(f"Auto-fix exhausted after {_MAX_FIX} attempts") from _compile_exc
                if _err == _prev_err:
                    _same_err_streak += 1
                else:
                    _same_err_streak = 0
                _prev_err = _err
                if _same_err_streak >= _MAX_SAME_STREAK:
                    log_fn("warning", "The same error persisted after several fix attempts. Try sending a chat message to fix it manually.")
                    raise _BuildFixExhausted(f"Auto-fix stalled — same error after {_same_err_streak} consecutive attempts") from _compile_exc
                retry_context = " (same error — trying a different approach)" if _same_err_streak > 0 else ""
                log_fn("info", f"Compilation failed — running auto-fix (attempt {_fix_attempt}){retry_context}…")
                logger.warning("Auto-fix triggered project=%s attempt=%d streak=%d: %s", project_id, _fix_attempt, _same_err_streak, _err[:200])
                _run_agent_fix(workspace.path, _err, template_key, app_name, blueprint, settings, log_fn, retry_hint=_same_err_streak > 0)
                workspace.sync_to_github(
                    commit_message="fix: auto-fix compilation errors",
                    push_url=push_url,
                )
                log_fn("info", "Fixes applied — retrying compilation…")

        preview_url: str | None = None
        artifact_url: str | None = None

        if artifact_path:
            log_fn("info", "Compilation successful — deploying preview…")

            if template_key == "flutter" and artifact_path.is_file() and settings.APPETIZE_API_TOKEN:
                preview_url = _deploy_appetize(artifact_path, settings.APPETIZE_API_TOKEN)
            elif template_key == "next" and artifact_path.is_dir():
                preview_url = _deploy_cloudflare_pages(artifact_path, app_name)
            elif template_key == "react_native" and artifact_path.is_dir():
                preview_url = _deploy_cloudflare_pages(artifact_path, app_name)

            if settings.CLOUDINARY_CLOUD_NAME:
                artifact_url = _upload_artifact(artifact_path, session_id)
        else:
            log_fn("warning", "Compilation produced no output — check the GitHub repo for errors.")

        now = datetime.now(timezone.utc)
        updates: dict = {"is_updating": False, "updated_at": now}
        if preview_url:
            updates["preview_url"] = preview_url
            log_fn("done", f"Preview deployed → {preview_url}")
        else:
            log_fn("warning", "Preview deployment skipped — no artifact or missing deploy credentials.", )
        if artifact_url:
            updates["artifact_url"] = artifact_url

        await db.collection("projects").document(project_id).update(updates)
        return {"preview_url": preview_url, "artifact_url": artifact_url}

    except _BuildFixExhausted as exc:
        user_msg = (
            "The preview couldn't be compiled automatically. "
            "Send a chat message like 'Fix the build errors' and the agent will try again."
        )
        logger.error("Preview auto-fix exhausted project=%s: %s", project_id, exc)
        log_fn("warning", user_msg)
        await db.collection("projects").document(project_id).update({
            "is_updating": False,
            "build_error": user_msg,
            "build_error_action": "user_fix",
            "updated_at": datetime.now(timezone.utc),
        })
        return {}

    except Exception as exc:
        from app.core.build_errors import sanitize_build_error
        build_err = sanitize_build_error(exc)
        logger.error("Preview build FAILED project=%s: %s", project_id, exc, exc_info=True)
        log_fn("error", f"Preview build failed: {build_err.message}")
        await db.collection("projects").document(project_id).update({
            "is_updating": False,
            "build_error": build_err.message,
            "build_error_action": build_err.action,
            "updated_at": datetime.now(timezone.utc),
        })
        raise
    finally:
        workspace.cleanup()


@celery_app.task(name="app.workers.build_worker.build_preview", bind=True, max_retries=0)
def build_preview(self, project_id: str, user_id: str) -> dict:
    """Celery entry point — manually triggered preview build."""
    logger.info("Preview build task started project=%s", project_id)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_preview(project_id, user_id))
    finally:
        loop.close()
