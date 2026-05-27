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
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
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

    # Blueprint generation backend: "claude" | "gemini" | "Qwen3"
    BP_MODEL: str = "claude"

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_API_KEY: str = ""

    # Recall.ai
    RECALL_API_KEY: str = ""
    RECALL_REGION: str = "us-east-1"
    RECALL_WORKSPACE_VERIFICATION_SECRET: str = ""
    PUBLIC_API_BASE_URL: str = ""

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

    # GitHub
    GITHUB_TOKEN: str = ""

    # Build templates (cloneable git repo URLs)
    TEMPLATE_FLUTTER: str = "https://github.com/seenickcode/flutter_starter"
    TEMPLATE_REACT_NATIVE: str = "https://github.com/obytes/react-native-template-obytes"
    TEMPLATE_NEXT: str = "https://github.com/ixartz/Next-js-Boilerplate"

    # Sentry
    SENTRY_DSN: str = ""

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
