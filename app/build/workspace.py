"""Workspace management — clone templates, git operations, and artifact builds."""
from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path("/tmp/forgefy_workspaces")


def _run(args: list[str], cwd: Path | None = None, timeout: int = 600) -> str:
    """Run a subprocess; return stdout; raise RuntimeError on failure."""
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


class Workspace:
    """Manages a cloned template directory for one session."""

    def __init__(self, session_id: uuid.UUID, template_key: str, template_url: str) -> None:
        self.session_id = session_id
        self.template_key = template_key
        self.template_url = template_url
        self.path = WORKSPACE_ROOT / str(session_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clone(self) -> None:
        """Clone the template (shallow) and strip .git so we start fresh."""
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.rmtree(self.path)

        logger.info("Cloning template=%s → %s", self.template_key, self.path)
        _run(["git", "clone", "--depth", "1", self.template_url, str(self.path)])
        shutil.rmtree(self.path / ".git", ignore_errors=True)
        logger.info("Workspace ready: %s", self.path)

    def init_git(self) -> None:
        """Initialise a fresh git repo inside the workspace."""
        _run(["git", "init"], cwd=self.path)
        _run(["git", "checkout", "-b", "main"], cwd=self.path)
        _run(["git", "config", "user.email", "build@forgefy.app"], cwd=self.path)
        _run(["git", "config", "user.name", "Forgefy Build"], cwd=self.path)

    def commit_all(self, message: str) -> None:
        """Stage all changes and commit."""
        _run(["git", "add", "-A"], cwd=self.path)
        _run(["git", "commit", "-m", message], cwd=self.path)

    def push(self, remote_url: str) -> None:
        """Add origin and push main."""
        _run(["git", "remote", "add", "origin", remote_url], cwd=self.path)
        _run(["git", "push", "-u", "origin", "main"], cwd=self.path)

    def cleanup(self) -> None:
        """Delete the workspace directory."""
        if self.path.exists():
            shutil.rmtree(self.path)
            logger.info("Workspace removed: %s", self.path)

    # ── Build ─────────────────────────────────────────────────────────────────


    def build_artifacts(self) -> Path | None:
        """Run the platform-appropriate build; return path to the artifact or None."""
        if self.template_key == "flutter":
            return self._build_flutter()
        if self.template_key == "react_native":
            return self._build_react_native()
        if self.template_key == "next":
            return self._build_next()
        return None

    def _build_flutter(self) -> Path | None:
        logger.info("Flutter: pub get + build apk (session=%s)", self.session_id)
        _run(["flutter", "pub", "get"], cwd=self.path, timeout=300)
        _run(
            ["flutter", "build", "apk", "--release", "--no-tree-shake-icons"],
            cwd=self.path,
            timeout=600,
        )
        apk = self.path / "build/app/outputs/flutter-apk/app-release.apk"
        if apk.exists():
            logger.info("APK ready: %s (%.1f MB)", apk, apk.stat().st_size / 1_048_576)
            return apk
        logger.warning("APK not found after flutter build")
        return None

    def _build_react_native(self) -> Path | None:
        logger.info("React Native: npm install + expo export (session=%s)", self.session_id)
        _run(["npm", "install", "--legacy-peer-deps"], cwd=self.path, timeout=300)

        # Try Expo web export first (no Android SDK needed)
        try:
            _run(["npx", "expo", "export", "--platform", "web"], cwd=self.path, timeout=300)
            dist = self.path / "dist"
            if dist.exists():
                logger.info("Expo web export ready: %s", dist)
                return dist
        except RuntimeError:
            pass

        # Fallback: bare React Native Android release build
        gradlew = self.path / "android" / "gradlew"
        if gradlew.exists():
            _run(["chmod", "+x", str(gradlew)])
            _run(
                ["./gradlew", "assembleRelease", "--no-daemon"],
                cwd=self.path / "android",
                timeout=600,
            )
            apk = self.path / "android/app/build/outputs/apk/release/app-release.apk"
            if apk.exists():
                logger.info("React Native APK ready: %s", apk)
                return apk

        logger.warning("No artifact produced for react_native build")
        return None

    def _build_next(self) -> Path | None:
        logger.info("Next.js: npm install + static export (session=%s)", self.session_id)
        _run(["npm", "install", "--legacy-peer-deps"], cwd=self.path, timeout=300)

        # Patch next.config.js/ts to add output: 'export' for Cloudflare Pages
        self._patch_next_config_for_export()

        _run(["npm", "run", "build"], cwd=self.path, timeout=300)

        # Static export lands in out/
        out = self.path / "out"
        if out.exists():
            logger.info("Next.js static export ready: %s", out)
            return out

        logger.warning("No out/ dir after next build — static export may have failed")
        return None

    def _patch_next_config_for_export(self) -> None:
        """Ensure next.config.js has output: 'export' so the build produces a static out/ dir."""
        for name in ("next.config.js", "next.config.ts", "next.config.mjs"):
            cfg = self.path / name
            if not cfg.exists():
                continue
            text = cfg.read_text(encoding="utf-8")
            if "output" in text:
                return  # already configured — respect what the agent wrote
            # Inject output: 'export' into the first nextConfig object literal
            patched = text.replace(
                "const nextConfig",
                "const nextConfig",
                1,
            )
            # Simple injection: add output before the closing brace of the config object
            patched = patched.replace(
                "module.exports = nextConfig",
                "nextConfig.output = 'export';\nmodule.exports = nextConfig",
            ).replace(
                "export default nextConfig",
                "nextConfig.output = 'export';\nexport default nextConfig",
            )
            cfg.write_text(patched, encoding="utf-8")
            logger.info("Patched %s with output: 'export'", name)
            return

        # No config file found — create a minimal one
        (self.path / "next.config.js").write_text(
            "/** @type {import('next').NextConfig} */\n"
            "const nextConfig = { output: 'export' };\n"
            "module.exports = nextConfig;\n",
            encoding="utf-8",
        )
        logger.info("Created next.config.js with output: 'export'")


# ---------------------------------------------------------------------------
# EditWorkspace — for prompt-driven updates on existing projects
# ---------------------------------------------------------------------------

class EditWorkspace:
    """Workspace for editing an existing project — cloned fresh from GitHub."""

    def __init__(self, project_id: uuid.UUID, repo_full_name: str, github_token: str) -> None:
        self.project_id = project_id
        self.repo_full_name = repo_full_name
        self.github_token = github_token
        self.path = WORKSPACE_ROOT / f"edit_{project_id}"

    def ensure(self) -> None:
        """Clone from GitHub if workspace doesn't exist; pull latest if it does."""
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        clone_url = f"https://{self.github_token}@github.com/{self.repo_full_name}.git"

        if self.path.exists():
            # Pull latest changes
            _run(["git", "remote", "set-url", "origin", clone_url], cwd=self.path)
            _run(["git", "pull", "--rebase", "origin", "main"], cwd=self.path, timeout=60)
            logger.info("Edit workspace updated via pull: %s", self.path)
        else:
            _run(["git", "clone", clone_url, str(self.path)])
            _run(["git", "config", "user.email", "build@forgefy.app"], cwd=self.path)
            _run(["git", "config", "user.name", "Forgefy Build"], cwd=self.path)
            logger.info("Edit workspace cloned: %s", self.path)

    def sync_to_github(self, commit_message: str, push_url: str) -> None:
        """Stage all changes, commit (if anything changed), and push."""
        _run(["git", "add", "-A"], cwd=self.path)
        # Check if there's anything staged
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.path, capture_output=True,
        )
        if diff.returncode == 0:
            logger.info("No changes to commit in edit workspace")
            return
        _run(["git", "commit", "-m", commit_message], cwd=self.path)
        _run(["git", "remote", "set-url", "origin", push_url], cwd=self.path)
        _run(["git", "push", "origin", "main"], cwd=self.path)
        logger.info("Synced edit workspace to GitHub: %s", self.repo_full_name)
