"""Minimal environment for subprocesses that execute model-authored project code.

The Celery worker's own environment holds ANTHROPIC_API_KEY, Firebase service
account credentials, GitHub tokens, Supabase keys and the database URL. Build
steps such as ``npm install``, ``npm test`` and ``flutter build`` execute scripts
out of a ``package.json`` / ``pubspec.yaml`` that the MODEL wrote, so passing the
inherited environment down hands every one of those secrets to generated code
that nobody reviewed.

Only variables a toolchain genuinely needs to find its own binaries, caches and
network egress are copied through. Everything else is dropped.
"""
from __future__ import annotations

import os

# Names are matched case-insensitively and re-emitted under the spelling found in
# the parent environment, because Windows normalises env keys to upper case while
# npm looks for the lower-case `npm_config_cache`.
_PASSTHROUGH: tuple[str, ...] = (
    # Interpreter / shell discovery
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    # Temp dirs — npm and gradle both fail hard without a writable one
    "TEMP",
    "TMP",
    "TMPDIR",
    # Locale: some toolchains emit mojibake or crash on a missing locale
    "LANG",
    "LC_ALL",
    # Package-manager caches. Dropping these does not break the build, it just
    # makes every build re-download the world.
    "NPM_CONFIG_CACHE",
    "NPM_CONFIG_PREFIX",
    "YARN_CACHE_FOLDER",
    "PNPM_HOME",
    "PUB_CACHE",
    "GRADLE_USER_HOME",
    # SDK roots
    "JAVA_HOME",
    "ANDROID_HOME",
    "ANDROID_SDK_ROOT",
    "FLUTTER_ROOT",
    # Egress configuration. Not secrets, and builds behind a corporate proxy or a
    # custom CA fail without them.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "GIT_SSL_CAINFO",
    "NODE_EXTRA_CA_CERTS",
)

_PASSTHROUGH_UPPER = frozenset(_PASSTHROUGH)


def build_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return a minimal environment dict for ``subprocess.run(env=...)``.

    ``extra`` is applied last, so a caller can add variables a specific command
    needs (for example ``GIT_TERMINAL_PROMPT=0``).
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _PASSTHROUGH_UPPER
    }
    # A missing PATH would make every `shutil.which` hit succeed and the spawn
    # then fail confusingly, so fall back to the OS default search path.
    if not any(k.upper() == "PATH" for k in env):
        env["PATH"] = os.defpath
    if extra:
        env.update(extra)
    return env
