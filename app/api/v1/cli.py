"""CLI-facing OpenAI-compatible chat endpoint.

The `forgefy` CLI provider (forgefy-cli's config.py BUILTINS) points its
base_url here and authenticates with a dashboard-minted `fgy_live_…` key —
the same keys, and the same monthly token bucket, as the developer extract
API (see app.deps.get_api_key, app/api/v1/extract.py). A user who signed up
on the Forgefy web app can `export FORGEFY_API_KEY=fgy_live_…` (minted on the
Developers page) and the CLI works against their account with no separate
CLI login.

Requests are forwarded to OpenRouter using the server's own
OPENROUTER_API_KEY — the CLI user never needs one — restricted to the same
curated, tool-calling-capable models the build agent already uses, so a CLI
caller can't spend the server's OpenRouter budget on an arbitrary model.
Unlike app/ai/openrouter.py's other callers, there is no chain fallback here:
the CLI lets its user pick one exact model and tells them so ("no automatic
paid fallback"), so this endpoint honours that and fails instead of silently
trying a different model.
"""
import logging
from contextlib import suppress
from functools import partial

import anyio
from fastapi import APIRouter, Request

from app.ai.openrouter import ASSISTANT, OpenRouterError, code_models, post_chat, resolve_models
from app.core.exceptions import ExternalServiceError, QuotaExceededError, ValidationError
from app.core.rate_limit import api_key_ident, limiter
from app.core.usage import evaluate_quota, record_usage
from app.deps import ApiKeyDep, DBSession, SettingsDep
from app.schemas.cli import CliChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _allowed_models() -> list[str]:
    """Curated, tool-calling-capable models the CLI may request."""
    return sorted(set(code_models()) | set(resolve_models(ASSISTANT)))


@router.get("/models")
@limiter.limit("60/minute", key_func=api_key_ident)
async def list_models(request: Request, api_key: ApiKeyDep) -> dict:
    """OpenAI-shaped model list — forgefy_cli.providers.ModelClient.models() reads `data[].id`."""
    return {"data": [{"id": model_id, "object": "model"} for model_id in _allowed_models()]}


@router.post("/chat/completions")
@limiter.limit("60/minute", key_func=api_key_ident)
async def chat_completions(
    request: Request,
    body: CliChatRequest,
    db: DBSession,
    settings: SettingsDep,
    api_key: ApiKeyDep,
) -> dict:
    """OpenAI-compatible chat completion, proxied to OpenRouter and metered like any build."""
    owner_id = str(api_key.owner_user_id)

    allowed = _allowed_models()
    if body.model not in allowed:
        raise ValidationError(
            f"Unknown or unsupported model '{body.model}'. Call GET /cli/models for the list."
        )

    outcome = await evaluate_quota(db, settings, owner_id)
    if outcome.action == "block":
        raise QuotaExceededError(outcome.message)
    # Same policy as builds/extract: a paid user over budget keeps working on
    # a free model instead of being hard-stopped mid-month. The CLI's own
    # "no automatic paid fallback" promise is about *provider* choice (never
    # silently switching to a different paid provider); this is the
    # platform's existing quota policy applying uniformly across every
    # surface, not the CLI's local fallback logic.
    model = resolve_models(ASSISTANT)[0] if outcome.action == "downgrade" else body.model

    router_key = (settings.OPENROUTER_API_KEY or "").strip()
    if not router_key:
        raise ExternalServiceError("The CLI backend is not configured on this server")

    messages = [m.model_dump(exclude_none=True) for m in body.messages]

    try:
        result = await anyio.to_thread.run_sync(
            partial(
                post_chat,
                model,
                messages,
                api_key=router_key,
                tools=body.tools,
                tool_choice=body.tool_choice,
                timeout=settings.OPENROUTER_TIMEOUT,
                referer=settings.OPENROUTER_SITE_URL,
                title=settings.OPENROUTER_APP_NAME,
            )
        )
    except OpenRouterError as exc:
        logger.warning("CLI chat failed key=%s model=%s: %s", api_key.id, model, exc)
        raise ExternalServiceError(str(exc)) from exc

    usage = result.get("usage") or {}
    total_tokens = int(usage.get("total_tokens") or 0)
    if total_tokens:
        with suppress(Exception):  # metering must never fail a served request
            await record_usage(db, owner_id, total_tokens)

    logger.info("CLI chat served key=%s model=%s tokens=%d", api_key.id, model, total_tokens)
    return result
