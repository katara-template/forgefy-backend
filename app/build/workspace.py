"""Workspace management — clone templates, git operations, and artifact builds."""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from app.build.subprocess_env import build_subprocess_env

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path("/tmp/forgefy_workspaces")

# Git authenticates over HTTPS by embedding the credential in the URL
# (https://<token>@github.com/...). Anything derived from such a command — an
# exception message, a log line, a Sentry event — carries a working credential
# unless it is scrubbed first.
# Credentials live in the userinfo slot: everything between "https://" and the
# first "@". `[^@]+` rather than `[^/\s@]+` — a malformed URL must not become a
# redaction bypass (a token containing a space used to slip past untouched).
_CREDENTIAL_IN_URL = re.compile(r"(https://)[^@]+@")


def _npm_install_args() -> list[str]:
    """Argv for `npm install`, optionally refusing package lifecycle scripts.

    `npm install` runs preinstall/postinstall hooks out of a package.json the
    model wrote, which is arbitrary code execution on the worker. --ignore-scripts
    closes that, but it also breaks any dependency that genuinely needs a
    postinstall step to fetch or compile a native binary (esbuild and sharp both
    do), so it is off unless an operator has confirmed their templates survive it.
    The scrubbed subprocess environment is the mitigation that always applies.
    """
    from app.config import get_settings

    args = ["npm", "install", "--legacy-peer-deps"]
    if get_settings().NPM_IGNORE_SCRIPTS:
        args.append("--ignore-scripts")
    return args


def _kill_workspace_jobs(path: Path) -> None:
    """Stop any background jobs the agent started in this workspace.

    Imported lazily and never allowed to raise: cleanup runs on the failure path
    too, and a bookkeeping error must not mask the original build error.
    """
    try:
        from app.build.jobs import kill_all_jobs

        kill_all_jobs(path)
    except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
        logger.warning("Could not stop background jobs for %s: %s", path, exc)


def _redact(text: str) -> str:
    """Strip credentials embedded in any https:// URL in `text`."""
    return _CREDENTIAL_IN_URL.sub(r"\1***@", text)


def _run(args: list[str], cwd: Path | None = None, timeout: int = 600) -> str:
    """Run a subprocess; return stdout; raise RuntimeError on failure.

    The raised message is redacted: `args` routinely contains a token-bearing
    clone/push URL, and this exception reaches the logs and Sentry.
    """
    # A scrubbed environment, not os.environ: this helper runs `npm install`,
    # `npm run build` and `flutter build` against a manifest the model wrote, and
    # the worker's own environment carries ANTHROPIC_API_KEY, Firebase
    # credentials, GitHub tokens and the database URL. Git keeps working because
    # its credentials travel in the remote URL, not the environment.
    env = build_subprocess_env({
        # Never prompt for git credentials — fail fast instead of hanging on missing /dev/tty
        "GIT_TERMINAL_PROMPT": "0",
    })
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
    )
    if result.returncode != 0:
        # Combine stdout + stderr: Flutter/Gradle print the Dart error on stdout
        combined = (result.stdout + "\n" + result.stderr).strip()
        # Redact both halves: git echoes the remote URL back in its own errors.
        raise RuntimeError(f"{_redact(' '.join(args))} failed:\n{_redact(combined)}")
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
        # MEMORY.md is written on a background thread; let it land first or it
        # misses this commit and is lost when the workspace is next re-cloned.
        from app.build.project_memory import flush_project_memory
        flush_project_memory()

        _run(["git", "add", "-A"], cwd=self.path)
        nothing_staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.path, capture_output=True, env=build_subprocess_env(),
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

    def sync_to_github(self, commit_message: str, push_url: str) -> bool:
        """Commit and push; no-op when the tree is clean. Returns True if pushed.

        Same shape as EditWorkspace.sync_to_github so WorkspaceAutoSync can drive
        either. No rebase here: this workspace owns a repo it just created, so
        nothing else is pushing to it concurrently.
        """
        if not self.commit_all(commit_message):
            return False
        self.push(push_url)
        return True

    def cleanup(self) -> None:
        """Delete the workspace directory."""
        # Kill first: a surviving `npm install` holds file handles inside the
        # directory we are about to remove, and keeps running against a path that
        # no longer exists.
        _kill_workspace_jobs(self.path)
        if self.path.exists():
            shutil.rmtree(self.path)
            logger.info("Workspace removed: %s", self.path)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_artifacts(self) -> Path | None:
        """Run the platform-appropriate build; return path to the artifact or None."""
        return build_artifacts_at(self.path, self.template_key)

    def write_supabase_env(self, url: str, anon_key: str) -> None:
        """Overwrite the build agent's placeholder Supabase values with real ones."""
        inject_supabase_env(self.path, self.template_key, url, anon_key)

    def write_neon_env(self, data_api_url: str) -> None:
        """Overwrite the build agent's placeholder Neon Data API URL with the real one."""
        inject_neon_env(self.path, self.template_key, data_api_url)

    def write_firebase_env(self, config: dict) -> None:
        """Overwrite the build agent's placeholder Firebase client config with real values."""
        inject_firebase_env(self.path, self.template_key, config)


# ---------------------------------------------------------------------------
# Shared artifact-build helpers (used by Workspace and EditWorkspace)
# ---------------------------------------------------------------------------

# Variable names the build agent is instructed to use (app/build/build_agent.py
# _BUILD_SUFFIX) — kept in one place so injection and the prompt can't drift apart.
SUPABASE_ENV_VAR_NAMES: dict[str, tuple[str, str]] = {
    "next": ("NEXT_PUBLIC_SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_ANON_KEY"),
    "react_native": ("EXPO_PUBLIC_SUPABASE_URL", "EXPO_PUBLIC_SUPABASE_ANON_KEY"),
    "flutter": ("SUPABASE_URL", "SUPABASE_ANON_KEY"),
}

NEON_ENV_VAR_NAMES: dict[str, str] = {
    "next": "NEXT_PUBLIC_NEON_DATA_API_URL",
    "react_native": "EXPO_PUBLIC_NEON_DATA_API_URL",
    "flutter": "NEON_DATA_API_URL",
}

# Logical config keys (as returned by app/integrations/firebase_management.py's
# get_web_app_config) mapped to the exact per-template env var name.
_FIREBASE_CONFIG_KEYS = (
    "apiKey", "authDomain", "projectId", "storageBucket", "messagingSenderId", "appId",
)

FIREBASE_ENV_VAR_NAMES: dict[str, dict[str, str]] = {
    "next": {
        "apiKey": "NEXT_PUBLIC_FIREBASE_API_KEY",
        "authDomain": "NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN",
        "projectId": "NEXT_PUBLIC_FIREBASE_PROJECT_ID",
        "storageBucket": "NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET",
        "messagingSenderId": "NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID",
        "appId": "NEXT_PUBLIC_FIREBASE_APP_ID",
    },
    "react_native": {
        "apiKey": "EXPO_PUBLIC_FIREBASE_API_KEY",
        "authDomain": "EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN",
        "projectId": "EXPO_PUBLIC_FIREBASE_PROJECT_ID",
        "storageBucket": "EXPO_PUBLIC_FIREBASE_STORAGE_BUCKET",
        "messagingSenderId": "EXPO_PUBLIC_FIREBASE_MESSAGING_SENDER_ID",
        "appId": "EXPO_PUBLIC_FIREBASE_APP_ID",
    },
    "flutter": {
        "apiKey": "FIREBASE_API_KEY",
        "authDomain": "FIREBASE_AUTH_DOMAIN",
        "projectId": "FIREBASE_PROJECT_ID",
        "storageBucket": "FIREBASE_STORAGE_BUCKET",
        "messagingSenderId": "FIREBASE_MESSAGING_SENDER_ID",
        "appId": "FIREBASE_APP_ID",
    },
}


def _inject_env_values(path: Path, values: dict[str, str]) -> None:
    """Overwrite/append the given KEY=value pairs in .env, preserving everything else."""
    env_path = path / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    replaced: set[str] = set()
    for i, line in enumerate(lines):
        name = line.split("=", 1)[0].strip()
        if name in values:
            lines[i] = f'{name}="{values[name]}"'
            replaced.add(name)

    for name, value in values.items():
        if name not in replaced:
            lines.append(f'{name}="{value}"')

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inject_supabase_env(path: Path, template_key: str, url: str, anon_key: str) -> None:
    """Overwrite the placeholder Supabase URL/anon key in .env with real values.

    .env (not .env.local) is intentional — the anon key is public-safe by
    design (protected by Row Level Security, not secrecy) and is committed to
    GitHub like every other NEXT_PUBLIC_-style value. Never call this with a
    service_role key or DB password — those never belong in a client app.
    """
    url_var, key_var = SUPABASE_ENV_VAR_NAMES.get(template_key, SUPABASE_ENV_VAR_NAMES["next"])
    _inject_env_values(path, {url_var: url, key_var: anon_key})
    ensure_forgefy_sdk_dependency(path, template_key)


def inject_neon_env(path: Path, template_key: str, data_api_url: str) -> None:
    """Overwrite the placeholder Neon Data API URL in .env with the real one.

    Only the Data API URL — never the raw Postgres connection string/password,
    which is a real secret and stays backend-only (see app/api/v1/projects.py).
    """
    url_var = NEON_ENV_VAR_NAMES.get(template_key, NEON_ENV_VAR_NAMES["next"])
    _inject_env_values(path, {url_var: data_api_url})
    ensure_forgefy_sdk_dependency(path, template_key)


def inject_firebase_env(path: Path, template_key: str, config: dict) -> None:
    """Overwrite the placeholder Firebase client config in .env with real values.

    `config` is the dict returned by firebase_management.get_web_app_config
    (apiKey/authDomain/projectId/storageBucket/messagingSenderId/appId) — all
    public-safe by design (security enforced by Firestore Security Rules).
    """
    var_names = FIREBASE_ENV_VAR_NAMES.get(template_key, FIREBASE_ENV_VAR_NAMES["next"])
    values = {
        var_names[key]: config[key]
        for key in _FIREBASE_CONFIG_KEYS
        if config.get(key) is not None
    }
    _inject_env_values(path, values)


# ---------------------------------------------------------------------------
# Forgefy client SDK — deterministic manifest backstop
# ---------------------------------------------------------------------------
# The build/update agent is *instructed* to add the Forgefy client SDK whenever
# a Supabase/Neon database is connected (see app/build/build_agent.py). This is
# the safety net so a build never breaks because the agent forgot: every time we
# inject real Supabase/Neon env values we also guarantee the dependency is in
# the manifest. Scoped to Supabase/Neon only — the SDK does not cover Firebase,
# so inject_firebase_env deliberately does NOT call this.

_FORGEFY_DART_DEP = ("forgefy_client", "^0.1.0")
_FORGEFY_NPM_DEP = ("@forgefy/client", "^0.1.0")


def ensure_forgefy_sdk_dependency(path: Path, template_key: str) -> None:
    """Guarantee the Forgefy client SDK is present in the app's manifest.

    Idempotent, and never overrides a version the agent already pinned. Best
    effort: any failure is logged, never raised, so it can't block a build.
    """
    try:
        if template_key == "flutter":
            _ensure_pubspec_dependency(path, *_FORGEFY_DART_DEP)
        else:  # next, react_native — both use package.json
            _ensure_package_json_dependency(_find_package_root(path), *_FORGEFY_NPM_DEP)
    except Exception as exc:  # a manifest tweak must never fail the build
        logger.warning("ensure_forgefy_sdk_dependency(%s) failed: %s", template_key, exc)


def _ensure_package_json_dependency(root: Path, pkg: str, version: str) -> None:
    import json

    manifest = root / "package.json"
    if not manifest.exists():
        logger.warning("No package.json under %s — cannot ensure %s", root, pkg)
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    deps = data.setdefault("dependencies", {})
    if pkg in deps or pkg in data.get("devDependencies", {}):
        return  # respect whatever the agent already declared
    deps[pkg] = version
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    logger.info("Ensured %s@%s in %s", pkg, version, manifest)


def _find_pubspec(path: Path) -> Path | None:
    """pubspec.yaml at the root, or one level down (mirrors _find_package_root)."""
    if (path / "pubspec.yaml").exists():
        return path / "pubspec.yaml"
    for child in sorted(path.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and (child / "pubspec.yaml").exists():
            return child / "pubspec.yaml"
    return None


def _ensure_pubspec_dependency(path: Path, pkg: str, version: str) -> None:
    pubspec = _find_pubspec(path)
    if pubspec is None:
        logger.warning("No pubspec.yaml under %s — cannot ensure %s", path, pkg)
        return
    lines = pubspec.read_text(encoding="utf-8").splitlines()
    if any(line.strip().startswith(f"{pkg}:") for line in lines):
        return  # already declared anywhere — respect it
    for i, line in enumerate(lines):
        # Top-level `dependencies:` block (column 0) — not dev_dependencies/overrides.
        if line.startswith("dependencies:"):
            lines.insert(i + 1, f"  {pkg}: {version}")
            pubspec.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Ensured %s: %s in %s", pkg, version, pubspec)
            return
    logger.warning("No top-level `dependencies:` block in %s — cannot ensure %s", pubspec, pkg)


# ---------------------------------------------------------------------------
# Forgefy UI SDK — deterministic manifest backstop (feature-flagged)
# ---------------------------------------------------------------------------
# forgefy_ui (Flutter) / @forgefy/ui (React web + native) apply to EVERY app of a
# template, not just DB-connected ones. Gated behind FORGEFY_UI_ENABLED because
# until the packages are published, force-adding them would break every build's
# pub-get / npm-install. Flip the env var on after publishing.

_FORGEFY_UI_DART_DEP = ("forgefy_ui", "^0.1.0")
_FORGEFY_UI_NPM_DEP = ("@forgefy/ui", "^0.1.0")


def _ui_enabled() -> bool:
    from app.config import get_settings
    return get_settings().FORGEFY_UI_ENABLED


def ensure_forgefy_ui_dependency(path: Path, template_key: str) -> None:
    """Force the Forgefy UI SDK into the manifest — no-op unless FORGEFY_UI_ENABLED
    is set. Best effort: never raises, so it can't block a build.
    """
    if not _ui_enabled():
        return
    try:
        if template_key == "flutter":
            _ensure_pubspec_dependency(path, *_FORGEFY_UI_DART_DEP)
        else:  # next, react_native
            _ensure_package_json_dependency(_find_package_root(path), *_FORGEFY_UI_NPM_DEP)
    except Exception as exc:  # a manifest tweak must never fail the build
        logger.warning("ensure_forgefy_ui_dependency(%s) failed: %s", template_key, exc)


# ---------------------------------------------------------------------------
# Forgefy signature — deterministic provenance / attribution banner
# ---------------------------------------------------------------------------
# Every app Forgefy builds carries a friendly, branded banner comment at the top
# of its entry file. Injected here (NOT via the LLM, which applies it
# inconsistently), idempotent (skips a file that already has it), and
# best-effort (never raises — a marketing banner must never break a build).

_SIGNATURE_MARKER = "Built with Forgefy"

# Banner body — no comment syntax; rendered with the right prefix per language.
# `//` covers Dart, TypeScript and JavaScript, i.e. every template we ship.
# Edit the copy/links here in one place.
_SIGNATURE_BODY = (
    "███████╗ ██████╗ ██████╗  ██████╗ ███████╗███████╗██╗   ██╗",
    "██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝██╔════╝╚██╗ ██╔╝",
    "█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  █████╗   ╚████╔╝ ",
    "██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ██╔══╝    ╚██╔╝  ",
    "██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗██║        ██║   ",
    "╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝        ╚═╝   ",
    "",
    "Built with Forgefy — from a single idea (or a meeting) to a real app.",
    "",
    "Build your own app   →  https://forgefy.app",
    "Docs & guides        →  https://forgefy.app/docs",
    "",
    "Made with care for makers who move fast.",
)

# Entry files to stamp, per template — checked in priority order; the first one
# that exists gets the banner (the app's main entry, not a registration stub).
_SIGNATURE_TARGETS: dict[str, tuple[str, ...]] = {
    "flutter": ("lib/main.dart",),
    "react_native": (
        "app/_layout.tsx", "app/_layout.js",  # Expo Router
        "src/app/_layout.tsx",
        "App.tsx", "App.js", "App.jsx",       # classic RN
        "index.tsx", "index.js",
    ),
    "next": (
        "app/layout.tsx", "app/layout.js",    # App Router
        "src/app/layout.tsx", "src/app/layout.js",
        "pages/_app.tsx", "pages/_app.js",    # Pages Router
        "src/pages/_app.tsx", "src/pages/_app.js",
    ),
}


def _render_signature(prefix: str = "//") -> str:
    return "\n".join(f"{prefix} {line}".rstrip() for line in _SIGNATURE_BODY)


def _is_leading_directive(line: str) -> bool:
    """True for a JS/TS module directive that must stay at the very top of the file."""
    s = line.strip().rstrip(";").strip().strip('"').strip("'")
    return s in ("use client", "use server", "use strict")


def _stamp_file(file: Path) -> bool:
    """Insert the banner at the top of one file. Returns True if it wrote."""
    text = file.read_text(encoding="utf-8")
    if _SIGNATURE_MARKER in text:
        return False  # already stamped — stay idempotent
    banner = _render_signature("//")
    lines = text.split("\n")
    # Keep a shebang or a leading "use client"/"use server" directive on line 1.
    insert_at = 0
    if lines:
        first = lines[0].strip()
        if first.startswith("#!") or _is_leading_directive(first):
            insert_at = 1
    new_lines = lines[:insert_at] + [banner, ""] + lines[insert_at:]
    file.write_text("\n".join(new_lines), encoding="utf-8")
    return True


def _signature_base(path: Path, template_key: str) -> Path:
    """Directory the entry-file paths are resolved against (handles subdir projects)."""
    if template_key == "flutter":
        pubspec = _find_pubspec(path)
        return pubspec.parent if pubspec else path
    return _find_package_root(path)


def stamp_forgefy_signature(path: Path, template_key: str) -> None:
    """Add the Forgefy provenance banner to the app's entry file.

    Deterministic, idempotent and best-effort: any failure is logged, never
    raised, so the banner can never block a build.
    """
    try:
        base = _signature_base(path, template_key)
        for rel in _SIGNATURE_TARGETS.get(template_key, ()):  # unknown template → no-op
            entry = base / rel
            if entry.is_file():
                wrote = _stamp_file(entry)
                logger.info(
                    "Forgefy signature %s → %s", "added" if wrote else "already present", entry
                )
                return
        logger.info("Forgefy signature: no entry file matched under %s (%s)", base, template_key)
    except Exception as exc:  # a banner must never fail the build
        logger.warning("stamp_forgefy_signature(%s) failed: %s", template_key, exc)


# ---------------------------------------------------------------------------
# Forgefy SEO metadata — stamped into deployed web builds
# ---------------------------------------------------------------------------
# Search engines index the deployed *.pages.dev HTML, so Forgefy provenance meta
# tags are injected into the BUILT HTML (not the source). This covers every web
# template uniformly (Next.js, React Native web) and can't be dropped by the LLM.
# Idempotent (marker-guarded) and best-effort — never raises, so it can't block
# a deploy, and is a no-op for non-web artifacts (e.g. a Flutter APK file).

_SEO_MARKER = "forgefy:seo"

_SEO_META_BLOCK = (
    "\n    <!-- forgefy:seo — provenance metadata added by Forgefy -->"
    '\n    <meta name="generator" content="Forgefy (https://forgefy.app)">'
    '\n    <meta name="forgefy" content="Built with Forgefy — https://forgefy.app">'
    '\n    <link rel="author" href="https://forgefy.app">\n  '
)


def inject_forgefy_seo_meta(build_dir: Path) -> int:
    """Insert Forgefy SEO meta tags into every HTML file's <head> in a web build.

    Returns the number of files stamped. Idempotent (skips files already carrying
    the marker) and best-effort — a no-op for a non-directory artifact (e.g. an
    APK) and never raises, so it can never block a deploy.
    """
    try:
        if not build_dir or not build_dir.is_dir():
            return 0
        stamped = 0
        for html in build_dir.rglob("*.html"):
            try:
                text = html.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # skip binary/unreadable files without failing the deploy
            if _SEO_MARKER in text or "</head>" not in text:
                continue  # already stamped, or no <head> to inject into
            html.write_text(text.replace("</head>", _SEO_META_BLOCK + "</head>", 1), encoding="utf-8")
            stamped += 1
        if stamped:
            logger.info("Injected Forgefy SEO meta into %d HTML file(s) under %s", stamped, build_dir)
        return stamped
    except Exception as exc:  # SEO meta must never block a deploy
        logger.warning("inject_forgefy_seo_meta(%s) failed: %s", build_dir, exc)
        return 0


_NEXT_CONFIG_CANONICAL = """\
/** @type {import('next').NextConfig} */
const nextConfig = {
  // Built for Cloudflare Pages via @cloudflare/next-on-pages.
  // Do NOT set output:'export' — that breaks API routes and SSR.
  images: { unoptimized: true },
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
};

module.exports = nextConfig;
"""

# Fallback config used when the Cloudflare adapter fails — static export only,
# no API routes, no SSR. Better than nothing.
_NEXT_CONFIG_STATIC_EXPORT = """\
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: { unoptimized: true },
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
};

module.exports = nextConfig;
"""


def _patch_next_config(path: Path) -> None:
    """Write a Cloudflare-compatible next.config (no output:'export').

    Always overwrites so the canonical flags are guaranteed to be present.
    """
    import re as _re
    for name in ("next.config.ts", "next.config.mjs", "next.config.js"):
        cfg = path / name
        if not cfg.exists():
            continue
        existing = cfg.read_text(encoding="utf-8").strip()
        # Already patched to our canonical form — nothing to do
        if "ignoreBuildErrors" in existing and "output:'export'" not in _re.sub(r"\s", "", existing):
            return
        cfg.write_text(_NEXT_CONFIG_CANONICAL, encoding="utf-8")
        logger.info("Patched %s for Cloudflare Pages (removed output:export)", name)
        return

    (path / "next.config.js").write_text(_NEXT_CONFIG_CANONICAL, encoding="utf-8")
    logger.info("Created next.config.js for Cloudflare Pages")


def _patch_next_config_static(path: Path) -> None:
    """Switch next.config to static export mode (fallback when Cloudflare adapter fails)."""
    for name in ("next.config.ts", "next.config.mjs", "next.config.js"):
        cfg = path / name
        if cfg.exists():
            cfg.write_text(_NEXT_CONFIG_STATIC_EXPORT, encoding="utf-8")
            logger.info("Switched %s to static-export fallback", name)
            return
    (path / "next.config.js").write_text(_NEXT_CONFIG_STATIC_EXPORT, encoding="utf-8")


def _ensure_wrangler_toml(path: Path, project_name: str) -> None:
    """Write wrangler.toml for @cloudflare/next-on-pages if it doesn't already exist."""
    import re as _re
    wrangler = path / "wrangler.toml"
    if wrangler.exists():
        return
    slug = _re.sub(r"[^a-z0-9-]", "-", project_name.lower()).strip("-")[:28] or "forgefy-app"
    wrangler.write_text(
        f'name = "{slug}"\n'
        'pages_build_output_dir = ".vercel/output/static"\n'
        'compatibility_date = "2025-01-01"\n'
        'compatibility_flags = ["nodejs_compat"]\n',
        encoding="utf-8",
    )
    logger.info("Created wrangler.toml for Cloudflare Pages project: %s", slug)


def _patch_tsconfig(path: Path) -> None:
    """Relax tsconfig strictness so minor AI-generated type errors don't block builds."""
    import json
    import re

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


def install_dependencies_at(path: Path, template_key: str, log_fn=None) -> bool:
    """Resolve the workspace's dependencies. Returns True on success.

    This has to happen BEFORE the agent phases run. A freshly cloned template has
    no node_modules and no resolved Dart packages, so the validator's analyze_code()
    tool ('dart analyze' / 'tsc --noEmit') reports every single import as
    unresolved — errors the validator then reads as real, and the fix pass tries to
    "fix" by rewriting working code.

    build_artifacts_at() installs too, but that runs after the agent has finished.
    Installing here is not duplicated work: the later install sees a warm tree and
    returns quickly.

    Never fatal — noisy analysis is worse than no build, but not worth failing over.
    """
    try:
        if template_key == "flutter":
            logger.info("Flutter: pub get (pre-agent) path=%s", path)
            _run(["flutter", "pub", "get"], cwd=path, timeout=300)
        elif template_key in ("next", "react_native"):
            root = _find_package_root(path)
            logger.info("npm install (pre-agent) path=%s", root)
            _run(_npm_install_args(), cwd=root, timeout=420)
        else:
            logger.warning("No dependency install defined for template=%s", template_key)
            return False
    except Exception as exc:
        logger.warning("Pre-agent dependency install failed (%s): %s", template_key, exc)
        if log_fn:
            log_fn(
                "warning",
                "Dependency install failed — static analysis may report import errors "
                "that are not real.",
            )
        return False

    if log_fn:
        log_fn("info", "Dependencies installed.")
    return True


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
        _run(_npm_install_args(), cwd=root, timeout=300)
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
        logger.info("Next.js: npm install + Cloudflare Pages build path=%s", root)
        _run(_npm_install_args(), cwd=root, timeout=300)
        _patch_next_config(root)
        _patch_tsconfig(root)

        # Attempt 1: @cloudflare/next-on-pages (supports SSR + API routes)
        try:
            _ensure_wrangler_toml(root, root.name)
            _run(
                ["npx", "--yes", "@cloudflare/next-on-pages"],
                cwd=root, timeout=600,
            )
            cf_out = root / ".vercel" / "output" / "static"
            if cf_out.exists():
                logger.info("Cloudflare Pages build succeeded → %s", cf_out)
                return cf_out
            logger.warning("@cloudflare/next-on-pages ran but .vercel/output/static not found")
        except RuntimeError as exc:
            logger.warning("@cloudflare/next-on-pages failed (%s) — falling back to static export", str(exc)[:200])

        # Attempt 2: static export fallback (no SSR / no API routes)
        logger.info("Next.js: falling back to next build with output:export")
        _patch_next_config_static(root)
        _run(["npm", "run", "build"], cwd=root, timeout=300)
        out = root / "out"
        if out.exists():
            logger.info("Static export fallback succeeded → %s", out)
            return out

        logger.warning("No artifact produced for next build (both approaches failed)")
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
        "*.tsbuildinfo",
        # .env is intentionally NOT gitignored — the agent only writes NEXT_PUBLIC_*,
        # Supabase anon key, Firebase client config, and other public-safe placeholders.
        # Truly secret keys go in .env.local which IS gitignored below.
        ".env.local",
        ".env*.local",
    ]

    # Bare .env patterns the template's .gitignore may ship that we strip so the
    # agent's .env (placeholder values only) gets committed and pushed to GitHub.
    _ENV_UNIGNORE = frozenset([".env", ".env.*", "*.env"])

    def _patch_gitignore(self) -> None:
        """Patch .gitignore: add required ignores and remove bare .env rules.

        Next.js (and most templates) ship with ".env" in their .gitignore, which
        silently drops the agent's .env from every push. We strip those lines so
        .env is committed (placeholder values only — safe to push), while keeping
        .env.local / .env*.local ignored because those hold real user secrets.
        """
        gitignore_path = self.path / ".gitignore"
        existing = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""

        cleaned, removed = [], 0
        for line in existing.splitlines():
            if line.strip() in self._ENV_UNIGNORE:
                removed += 1
            else:
                cleaned.append(line)
        existing = "\n".join(cleaned) + ("\n" if cleaned else "")

        missing = [p for p in self._REQUIRED_IGNORES if p not in existing]
        if removed or missing:
            with gitignore_path.open("w", encoding="utf-8") as f:
                f.write(existing)
                if missing:
                    f.write("\n# --- build outputs (auto-patched by Forgefy) ---\n")
                    f.write("\n".join(missing) + "\n")
            logger.info(
                "Patched .gitignore: removed %d .env rule(s), added %d missing entries",
                removed, len(missing),
            )

    def sync_to_github(self, commit_message: str, push_url: str) -> bool:
        """Stage all changes, commit, rebase onto remote, then push.

        Handles the case where the remote is ahead of local (e.g. user pushed
        directly to GitHub between when we cloned and when the agent finished).
        Agent changes always win on conflict — the agent's code is the canonical
        output for this update.
        """
        # Patch .gitignore first (strips bare .env rules, adds required ignores).
        self._patch_gitignore()
        # If .env was previously cached as ignored, untrack it so git add -A
        # can now stage it as a regular file.
        subprocess.run(
            ["git", "rm", "--cached", "--ignore-unmatch", ".env"],
            cwd=self.path, capture_output=True, env=build_subprocess_env(),
        )
        _run(["git", "add", "-A"], cwd=self.path)
        nothing_staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self.path, capture_output=True, env=build_subprocess_env(),
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
                cwd=self.path, capture_output=True, env=build_subprocess_env(),
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

    def write_supabase_env(self, template_key: str, url: str, anon_key: str) -> None:
        """Overwrite the build agent's placeholder Supabase values with real ones."""
        inject_supabase_env(self.path, template_key, url, anon_key)

    def write_neon_env(self, template_key: str, data_api_url: str) -> None:
        """Overwrite the build agent's placeholder Neon Data API URL with the real one."""
        inject_neon_env(self.path, template_key, data_api_url)

    def write_firebase_env(self, template_key: str, config: dict) -> None:
        """Overwrite the build agent's placeholder Firebase client config with real values."""
        inject_firebase_env(self.path, template_key, config)

    def cleanup(self) -> None:
        """Delete the workspace directory to free disk space."""
        _kill_workspace_jobs(self.path)
        if self.path.exists():
            shutil.rmtree(self.path)
            logger.info("Edit workspace removed: %s", self.path)
