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

    # Deepgram
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_MODEL: str = "nova-3"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Ollama (local Qwen3)
    OLLAMA_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_TIMEOUT: int = 300

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
