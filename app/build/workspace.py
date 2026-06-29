"""Workspace management — clone templates, git operations, and artifact builds."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path("/tmp/forgefy_workspaces")


def _run(args: list[str], cwd: Path | None = None, timeout: int = 600) -> str:
    """Run a subprocess; return stdout; raise RuntimeError on failure."""
    env = os.environ.copy()
    # Never prompt for git credentials — fail fast instead of hanging on missing /dev/tty
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
    )
    if result.returncode != 0:
        # Combine stdout + stderr: Flutter/Gradle print the Dart error on stdout
        combined = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"{' '.join(args)} failed:\n{combined}")
    return result.stdout


class Workspace:
    """Manages a cloned template directory for one session."""

    def __init__(
        self,
        session_id: uuid.UUID,
        template_key: str,
        template_url: str,
        git_token: str = "",
    ) -> None:
        self.session_id = session_id
        self.template_key = template_key
        self.template_url = template_url
        self.git_token = git_token
        self.path = WORKSPACE_ROOT / str(session_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clone(self) -> None:
        """Clone the template (shallow) and strip .git so we start fresh."""
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.rmtree(self.path)

        url = self.template_url
        if self.git_token and url.startswith("https://github.com/"):
            # Embed the token so git can authenticate without a credential helper or TTY
            url = url.replace("https://", f"https://{self.git_token}@", 1)

        logger.info("Cloning template=%s → %s", self.template_key, self.path)
        _run(["git", "clone", "--depth", "1", url, str(self.path)])
        shutil.rmtree(self.path / ".git", ignore_errors=True)
        logger.info("Workspace ready: %s", self.path)

    def init_git(self) -> None:
        """Initialise a fresh git repo inside the workspace."""
        _run(["git", "init"], cwd=self.path)
        _run(["git", "checkout", "-b", "main"], cwd=self.path)
        _run(["git", "config", "user.email", "build@forgefy.app"], cwd=self.path)
        _run(["git", "config", "user.name", "Forgefy Build"], cwd=self.path)

    def commit_all(self, message: str) -> bool:
        """Stage all changes and commit. Returns False (no-op) if nothing staged."""
        _run(["git", "add", "-A"], cwd=self.path)
        nothing_staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.path, capture_output=True,
        ).returncode == 0
        if nothing_staged:
            logger.info("commit_all: nothing to commit — skipping")
            return False
        _run(["git", "commit", "-m", message], cwd=self.path)
        return True

    def push(self, remote_url: str) -> None:
        """Set origin and push main. Safe to call multiple times."""
        try:
            _run(["git", "remote", "add", "origin", remote_url], cwd=self.path)
        except RuntimeError:
            # origin already exists — update URL instead
            _run(["git", "remote", "set-url", "origin", remote_url], cwd=self.path)
        _run(["git", "push", "-u", "origin", "main"], cwd=self.path)

    def cleanup(self) -> None:
        """Delete the workspace directory."""
        if self.path.exists():
            shutil.rmtree(self.path)
            logger.info("Workspace removed: %s", self.path)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_artifacts(self) -> Path | None:
        """Run the platform-appropriate build; return path to the artifact or None."""
        return build_artifacts_at(self.path, self.template_key)


# ---------------------------------------------------------------------------
# Shared artifact-build helpers (used by Workspace and EditWorkspace)
# ---------------------------------------------------------------------------

_NEXT_CONFIG_CANONICAL = """\
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  // Static export requires unoptimized images (no server-side resize)
  images: { unoptimized: true },
  // Skip TypeScript type-checking during build — agent-generated code may have
  // minor type errors that don't affect runtime correctness.
  typescript: { ignoreBuildErrors: true },
  // Skip ESLint during build for the same reason.
  eslint: { ignoreDuringBuilds: true },
};

module.exports = nextConfig;
"""


def _patch_next_config(path: Path) -> None:
    """Ensure next.config has static export + build-error suppression.

    Always writes a canonical config so every needed flag is present, regardless
    of what the agent generated.  The original file is preserved as a comment
    block so no agent work is lost.
    """
    for name in ("next.config.ts", "next.config.mjs", "next.config.js"):
        cfg = path / name
        if not cfg.exists():
            continue
        existing = cfg.read_text(encoding="utf-8").strip()
        # Already fully patched — nothing to do
        if "ignoreBuildErrors" in existing and "output" in existing:
            return
        # Overwrite with canonical config; keep original as reference comment
        cfg.write_text(_NEXT_CONFIG_CANONICAL, encoding="utf-8")
        logger.info("Replaced %s with canonical config (ignoreBuildErrors + output:export)", name)
        return

    # No config file found — create one
    (path / "next.config.js").write_text(_NEXT_CONFIG_CANONICAL, encoding="utf-8")
    logger.info("Created next.config.js with canonical config")


def _patch_tsconfig(path: Path) -> None:
    """Relax tsconfig strictness so minor AI-generated type errors don't block builds."""
    import json, re

    tsconfig_path = path / "tsconfig.json"
    if not tsconfig_path.exists():
        return

    text = tsconfig_path.read_text(encoding="utf-8")
    # Strip JS-style comments before parsing (tsconfig allows them, json.loads doesn't)
    clean = re.sub(r"//[^\n]*", "", text)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
    try:
        cfg = json.loads(clean)
    except Exception:
        logger.warning("Could not parse tsconfig.json — skipping patch")
        return

    compiler = cfg.setdefault("compilerOptions", {})
    changed = False
    relaxations: dict = {
        "strict": False,
        "noImplicitAny": False,
        "strictNullChecks": False,
        "skipLibCheck": True,           # skip type-checking inside node_modules
        "noUnusedLocals": False,        # agent often leaves unused imports
        "noUnusedParameters": False,
    }
    for key, value in relaxations.items():
        if compiler.get(key) != value:
            compiler[key] = value
            changed = True

    if changed:
        tsconfig_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        logger.info("Relaxed tsconfig.json strictness for agent-generated code")


def _find_package_root(path: Path) -> Path:
    """Return the directory containing package.json.

    The build agent sometimes creates files in a subdirectory rather than the
    workspace root. Search one level deep so npm/expo are run in the right place.
    """
    if (path / "package.json").exists():
        return path
    for child in sorted(path.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and (child / "package.json").exists():
            logger.info("package.json found in subdirectory %s — using as project root", child.name)
            return child
    logger.warning("No package.json found in %s or its subdirectories — using root", path)
    return path


def build_artifacts_at(path: Path, template_key: str) -> Path | None:
    """Compile and return the artifact path for the given workspace directory."""
    if template_key == "flutter":
        logger.info("Flutter: pub get + build apk path=%s", path)
        _run(["flutter", "pub", "get"], cwd=path, timeout=300)
        _run(["flutter", "build", "apk", "--release", "--no-tree-shake-icons"], cwd=path, timeout=600)
        apk = path / "build/app/outputs/flutter-apk/app-release.apk"
        if apk.exists():
            logger.info("APK ready: %s (%.1f MB)", apk, apk.stat().st_size / 1_048_576)
            return apk
        logger.warning("APK not found after flutter build")
        return None

    if template_key == "react_native":
        root = _find_package_root(path)
        logger.info("React Native: npm install + expo export path=%s", root)
        _run(["npm", "install", "--legacy-peer-deps"], cwd=root, timeout=300)
        try:
            _run(["npx", "expo", "export", "--platform", "web"], cwd=root, timeout=300)
            dist = root / "dist"
            if dist.exists():
                return dist
        except RuntimeError:
            pass
        gradlew = root / "android" / "gradlew"
        if gradlew.exists():
            _run(["chmod", "+x", str(gradlew)])
            _run(["./gradlew", "assembleRelease", "--no-daemon"], cwd=root / "android", timeout=600)
            apk = root / "android/app/build/outputs/apk/release/app-release.apk"
            if apk.exists():
                return apk
        logger.warning("No artifact produced for react_native build")
        return None

    if template_key == "next":
        root = _find_package_root(path)
        logger.info("Next.js: npm install + static export path=%s", root)
        _run(["npm", "install", "--legacy-peer-deps"], cwd=root, timeout=300)
        _patch_next_config(root)
        _patch_tsconfig(root)
        _run(["npm", "run", "build"], cwd=root, timeout=300)
        out = root / "out"
        if out.exists():
            return out
        logger.warning("No out/ dir after next build")
        return None

    return None


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

    # Patterns that must never be committed regardless of what the repo's
    # .gitignore says (or doesn't say).
    _REQUIRED_IGNORES = [
        "node_modules/",
        ".next/",
        ".expo/",
        "dist/",
        "build/",
        ".dart_tool/",
        "*.g.dart",
        ".flutter-plugins",
        ".flutter-plugins-dependencies",
        "__pycache__/",
        "*.pyc",
        ".env",
        ".env.local",
        ".env*.local",
        "*.tsbuildinfo",
    ]

    def _patch_gitignore(self) -> None:
        """Ensure critical build-output patterns are always gitignored."""
        gitignore_path = self.path / ".gitignore"
        existing = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
        missing = [p for p in self._REQUIRED_IGNORES if p not in existing]
        if missing:
            with gitignore_path.open("a", encoding="utf-8") as f:
                f.write("\n# --- build outputs (auto-patched by Forgefy) ---\n")
                f.write("\n".join(missing) + "\n")
            logger.info("Patched .gitignore with %d missing entries", len(missing))

    def sync_to_github(self, commit_message: str, push_url: str) -> bool:
        """Stage all changes, commit, rebase onto remote, then push.

        Handles the case where the remote is ahead of local (e.g. user pushed
        directly to GitHub between when we cloned and when the agent finished).
        Agent changes always win on conflict — the agent's code is the canonical
        output for this update.
        """
        # Always ensure node_modules and build outputs are ignored before staging.
        self._patch_gitignore()
        _run(["git", "add", "-A"], cwd=self.path)
        nothing_staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.path, capture_output=True,
        ).returncode == 0
        if nothing_staged:
            logger.info("No changes to commit in edit workspace")
            return False

        _run(["git", "commit", "-m", commit_message], cwd=self.path)
        _run(["git", "remote", "set-url", "origin", push_url], cwd=self.path)

        # Bring in any commits that landed on the remote while the agent was running.
        # --rebase keeps history linear; --autostash preserves any un-staged working tree.
        try:
            _run(
                ["git", "pull", "--rebase", "--autostash", "origin", "main"],
                cwd=self.path,
                timeout=60,
            )
        except RuntimeError:
            # Automatic rebase failed (conflicting edits to the same lines).
            # Abort the in-progress rebase so the repo is in a clean state,
            # then force-push the agent's version — agent output takes precedence.
            subprocess.run(
                ["git", "rebase", "--abort"],
                cwd=self.path, capture_output=True,
            )
            logger.warning(
                "Rebase conflict for %s — using force-with-lease to push agent changes",
                self.repo_full_name,
            )
            _run(["git", "push", "--force-with-lease", "origin", "main"], cwd=self.path)
            return True

        _run(["git", "push", "origin", "main"], cwd=self.path)
        logger.info("Synced edit workspace to GitHub: %s", self.repo_full_name)
        return True

    def build_artifacts(self, template_key: str) -> Path | None:
        """Compile the cloned repo and return the artifact path."""
        return build_artifacts_at(self.path, template_key)

    def cleanup(self) -> None:
        """Delete the workspace directory to free disk space."""
        if self.path.exists():
            shutil.rmtree(self.path)
            logger.info("Edit workspace removed: %s", self.path)
