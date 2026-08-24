"""Application settings loaded from environment / .env file."""
import ast
import json
from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    APP_ENV: str = "development"
    SECRET_KEY: str = "changeme"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8 hours — long enough to cover a full meeting-to-build flow
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    CORS_ORIGINS: Any = ["http://localhost:3000"]

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── Build queue admission control ────────────────────────────────────────
    # Builds take minutes each, so the queue cannot absorb a spike. Past this
    # backlog new build requests are rejected with 429 rather than accepted into
    # a queue that would grow until Redis OOMs. 0 disables the check.
    BUILD_QUEUE_MAX_DEPTH: int = 500
    # Redis re-delivers an acks_late task that outlives this window. It MUST
    # exceed the longest possible build or a slow build is handed to a second
    # worker while the first still runs — two agents writing one workspace.
    CELERY_VISIBILITY_TIMEOUT: int = 14400  # 4h, well past the ~22min worst case
    # Recycle worker processes periodically: agent contexts are large and a
    # long-lived prefork child accumulates memory it never returns.
    CELERY_MAX_TASKS_PER_CHILD: int = 50
    # Concurrent tasks per worker container. Set here rather than as a CLI flag
    # so docker-compose and Dockerfile.worker cannot drift apart — a --concurrency
    # argument overrides this, so neither entrypoint passes one. Kept low on
    # purpose: a build holds a workspace, a toolchain and a large agent context,
    # so add capacity with more worker containers, not more processes per box.
    CELERY_CONCURRENCY: int = 2

    # Deepgram
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_MODEL: str = "nova-3"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Ollama (Qwen3 via a local daemon, or the hosted service at ollama.com)
    OLLAMA_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_TIMEOUT: int = 300
    # Set this to use Ollama Cloud instead of a local daemon: requests gain an
    # Authorization: Bearer header and OLLAMA_URL defaults to https://ollama.com
    # (see app/ai/ollama_http.py). Cloud model tags end in '-cloud', so pair it
    # with e.g. OLLAMA_MODEL=qwen3.5:cloud.
    OLLAMA_API_KEY: str = ""
    # Optional separate model for code generation. Blueprint work rewards careful
    # extraction while build work rewards fast, decisive tool use, and the best
    # model for one is often the worst for the other. Blank = use OLLAMA_MODEL. kimi-k2.7-code, qwen3.5:397b, deepseek-v4-pro, deepseek-v4-flash, glm-5.1, glm-5.2, kimi-k2.6, kimi-k3, minimax-m2.7, mistral-large-3:675b
    OLLAMA_BUILD_MODEL: str = ""

    # OpenRouter — the hosted backend for the "Qwen3" setting. When
    # OPENROUTER_API_KEY is set, BP_MODEL/BUILD_MODEL="Qwen3" routes each action
    # to the model best suited to it (app/ai/openrouter.py) instead of hitting a
    # local Ollama. Leave the key blank to keep using local Ollama.
    OPENROUTER_API_KEY: str = ""
    # Pin one model for every action, bypassing per-action routing. Mostly for
    # debugging — e.g. "qwen/qwen3-coder:free".
    OPENROUTER_MODEL: str = ""
    # Allow paid models as fallbacks when the free tier is rate-limited.
    OPENROUTER_ALLOW_PAID: bool = False
    OPENROUTER_TIMEOUT: int = 180
    # Optional attribution shown on OpenRouter's dashboard/leaderboards.
    OPENROUTER_SITE_URL: str = "https://forgefy.app"
    OPENROUTER_APP_NAME: str = "Forgefy"

    # Blueprint generation backend (extraction + synthesis): "claude" | "gemini" | "Qwen3"
    BP_MODEL: str = "claude"
    # App build backend (code generation): "claude" | "gemini" | "Qwen3" | "gpt"
    BUILD_MODEL: str = "gemini"

    # Pass --ignore-scripts to `npm install` in build workspaces. This stops
    # preinstall/postinstall hooks in a model-authored package.json from running
    # arbitrary code on the worker, but it also breaks dependencies that need a
    # postinstall step to fetch or compile a native binary (esbuild, sharp).
    # Off by default; turn it on once your templates are confirmed to install
    # cleanly without lifecycle scripts.
    NPM_IGNORE_SCRIPTS: bool = False

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Recall.ai
    RECALL_API_KEY: str = ""
    RECALL_REGION: str = "us-east-1"
    RECALL_WORKSPACE_VERIFICATION_SECRET: str = ""
    PUBLIC_API_BASE_URL: str = ""
    # JPEG shown as the bot's camera feed in meetings (its "avatar").
    # Relative paths resolve against the backend root. Skipped if missing.
    RECALL_BOT_AVATAR_PATH: str = "assets/bot_avatar.jpg"

    # ── Zoom bot provider ────────────────────────────────────────────────────
    # Which bot joins Zoom meetings: "recall" (cloud, billed per bot-hour) or
    # "self_hosted" (our own Meeting SDK container — see zoom-bot/README.md).
    # Affects Zoom only; Meet and Teams always use Recall.
    ZOOM_BOT_PROVIDER: str = "recall"

    # Meeting SDK app credentials from marketplace.zoom.us. The secret is used
    # only to sign the bot's JWT — it never enters the bot container, which
    # joins untrusted meetings.
    ZOOM_SDK_CLIENT_ID: str = ""
    ZOOM_SDK_CLIENT_SECRET: str = ""

    # OAuth credentials for minting per-meeting OBF tokens, which Zoom has
    # required since 2026-03-02 to join meetings hosted on other accounts.
    # Leave blank when the Marketplace app provides both features under one
    # Client ID — the ZOOM_SDK_* values are used as the fallback.
    ZOOM_OAUTH_CLIENT_ID: str = ""
    ZOOM_OAUTH_CLIENT_SECRET: str = ""

    # Image each per-meeting container is launched from. Build it with
    # `docker compose --profile build-only build zoom-bot`.
    ZOOM_BOT_IMAGE: str = "forgefy-zoom-bot:latest"

    # Docker network the bot joins. It must be able to reach the API by name;
    # blank means Docker's default bridge, which usually cannot.
    ZOOM_BOT_NETWORK: str = ""

    # Where the bot posts transcript and lifecycle events. Internal by default:
    # unlike Recall this needs no publicly reachable URL and no tunnel.
    ZOOM_BOT_CALLBACK_URL: str = "http://api:5000"

    # Shown in the participant list — the bot's primary disclosure to everyone
    # in the meeting, so keep it self-describing.
    ZOOM_BOT_DISPLAY_NAME: str = "Forgefy Notetaker"

    # Require the host to grant local recording privilege before any audio is
    # captured. Zoom enforces this regardless; setting it false only stops the
    # bot from *asking*, which then requires a pre-authorized join token.
    ZOOM_BOT_REQUIRE_HOST_CONSENT: bool = True

    # Leave once no other participant remains, so an abandoned meeting cannot
    # hold a container open indefinitely. 0 disables the watchdog.
    ZOOM_BOT_LEAVE_AFTER_SILENCE_SECS: int = 120

    @field_validator("ZOOM_BOT_PROVIDER")
    @classmethod
    def _known_zoom_provider(cls, v: str) -> str:
        """Fail at startup rather than at the first meeting on a typo."""
        allowed = {"recall", "self_hosted"}
        value = v.strip().lower()
        if value not in allowed:
            raise ValueError(
                f"ZOOM_BOT_PROVIDER must be one of {sorted(allowed)}, got {v!r}"
            )
        return value

    # Cloudinary (APK / bundle storage)
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""

    # Cloudflare Pages (web preview deployments)
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""

    # Appetize.io (Flutter APK browser preview)
    APPETIZE_API_TOKEN: str = ""

    # GitHub OAuth App (for linking user personal GitHub)
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""
    FRONTEND_URL: str = "http://localhost:3000"

    # Supabase OAuth App (for linking user Supabase account + provisioning DBs)
    SUPABASE_CLIENT_ID: str = ""
    SUPABASE_CLIENT_SECRET: str = ""

    # Neon — embedded/project-per-user model: one platform-level API key,
    # no per-user OAuth. Projects are provisioned under Forgefy's own account.
    # NEON_ORG_ID is required for organization-scoped API keys (Neon console →
    # organization settings page, looks like "org-...").
    NEON_API_KEY: str = ""
    NEON_ORG_ID: str = ""

    # Firebase OAuth App (Google Cloud OAuth client) — for linking a user's own
    # Google account so Forgefy can provision a Firebase/Firestore project under
    # IT, not Forgefy's own. Distinct from FIREBASE_CREDENTIALS_JSON/PATH (raw
    # env vars read in app/db/firebase.py), which are Forgefy's own service
    # account for its own Firestore usage.
    FIREBASE_OAUTH_CLIENT_ID: str = ""
    FIREBASE_OAUTH_CLIENT_SECRET: str = ""

    # Firebase Web API key (public-safe, from the Firebase console's project
    # settings) — needed server-side to verify email/password logins against
    # the Identity Toolkit REST API; the Admin SDK cannot check passwords.
    FIREBASE_WEB_API_KEY: str = ""

    # Encryption key for secrets stored at rest (OAuth tokens, DB passwords).
    # Any string works — it's hashed into a Fernet key. Change in production.
    SECRETS_ENCRYPTION_KEY: str = "changeme-replace-with-a-real-secret"

    # GitHub
    GITHUB_TOKEN: str = ""

    # Build templates (cloneable git repo URLs)
    TEMPLATE_FLUTTER: str = "https://github.com/katara-template/flutter-clean-bloc.git"
    TEMPLATE_REACT_NATIVE: str = "https://github.com/katara-template/rn-redux.git"
    TEMPLATE_NEXT: str = "https://github.com/katara-template/next-ts.git"

    # ElevenLabs Conversational AI (voice chat)
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_AGENT_ID: str = ""
    # Shared secret used to sign voice session tokens so the /voice/llm
    # endpoint can verify calls originated from a valid signed session.
    ELEVENLABS_VOICE_SECRET: str = "changeme-voice-secret"

    # fal.ai (image + video generation)
    FAL_API_KEY: str = ""

    # Notchpay (payments — card, MTN MoMo, Orange Money)
    NOTCHPAY_PUBLIC_KEY: str = ""   # pk_... from Notchpay dashboard
    NOTCHPAY_SECRET_HASH: str = ""  # hash secret for webhook verification

    # Sentry
    SENTRY_DSN: str = ""

    # Forgefy UI SDK — when true, the build/update agents use forgefy_ui (Flutter)
    # and @forgefy/ui (React web + native) for layout/animation and force those
    # packages into generated manifests. Keep false until they are published to
    # pub.dev / npm, else every generated build fails at pub-get / npm-install.
    FORGEFY_UI_ENABLED: bool = False

    @field_validator("REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", mode="before")
    @classmethod
    def fix_rediss_ssl(cls, v: Any) -> Any:
        """Ensure rediss:// URLs carry ssl_cert_reqs=none (redis-py canonical form).

        redis-py only accepts lowercase none/optional/required as the value for
        ssl_cert_reqs in URL query strings.  The old value 'CERT_NONE' raises
        RedisError: Invalid SSL Certificate Requirements Flag: CERT_NONE.
        Cloud providers (Render, Railway, Upstash) give bare rediss:// URLs
        without any cert flag, so we append the correct value here.
        """
        if not isinstance(v, str) or not v.startswith("rediss://"):
            return v
        # Normalise an existing CERT_NONE (uppercase) left from the old code
        if "ssl_cert_reqs=CERT_NONE" in v:
            return v.replace("ssl_cert_reqs=CERT_NONE", "ssl_cert_reqs=none")
        if "ssl_cert_reqs" not in v:
            sep = "&" if "?" in v else "?"
            return f"{v}{sep}ssl_cert_reqs=none"
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Accept a JSON array, a Python-style list, or a comma-separated string."""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("["):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    # Fallback: handles single-quoted lists like ['url1', 'url2']
                    result = ast.literal_eval(stripped)
                    if isinstance(result, list):
                        return [str(x) for x in result]
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
