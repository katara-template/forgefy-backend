"""Runtime build-model resolution and the admin-curated model catalogue.

Which model actually *runs* a build resolves, per user, in priority order:

    per-user override  >  Firestore ``system/config.build_model``  >  ``.env BUILD_MODEL``

The *catalogue of models offered to users* for selection is admin-curated from
the dashboard and stored under the same ``system/config`` document as a
``build_models`` array (one entry per model, carrying the metadata UIs render).
Until an operator curates it, ``DEFAULT_BUILD_MODELS`` — the models the worker
ships adapters for — is what every user sees.

New models can be added to the catalogue from the admin app; doing so makes them
selectable by users and assignable as overrides. The build worker
(``app.build.build_agent._agent_adapter``) only knows how to drive the models in
``VALID_BUILD_MODELS``; a catalogue entry outside that set is still offered to
users and accepted by the API, but a build that resolves to it falls back to the
configured default model. Surfacing a brand-new provider therefore requires only
adding it here — no frontend deploy.

Providers are used *directly* wherever the vendor offers its own key: ``claude``
→ Anthropic, ``gemini`` → Google, ``gpt`` → OpenAI, ``deepseek`` → DeepSeek's
own ``api.deepseek.com``. ``Qwen3`` is the only routed backend (Ollama, or
OpenRouter when ``OPENROUTER_API_KEY`` is set).

Most keys name a *provider*; the model within it comes from the matching
``*_MODEL`` setting. DeepSeek is the exception, because its two live models
differ enough to be worth choosing between: ``deepseek`` uses ``DEEPSEEK_MODEL``,
while ``deepseek:<model-id>`` pins one model outright. That form is validated by
shape rather than against ``VALID_BUILD_MODELS``, so a newly released DeepSeek
model can be added to the catalogue and selected immediately, with no deploy.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# The build-model identifiers the worker can map to a provider adapter
# (see ``app.build.build_agent._agent_adapter``). Kept as a flat tuple of keys
# so existing imports and tests that compare against it keep working.
VALID_BUILD_MODELS = ("claude", "Qwen3", "gemini", "gpt", "deepseek")

# DeepSeek is selectable *per model*. The bare key uses the configured
# DEEPSEEK_MODEL; a ``deepseek:<model-id>`` key names an explicit model, so every
# model DeepSeek ships — and every future one — can be picked from the model
# selector (or added by an admin from the dashboard) with no code change. Both
# spellings reach the same backend and the same DEEPSEEK_API_KEY.
DEEPSEEK_KEY = "deepseek"
DEEPSEEK_KEY_PREFIX = "deepseek:"


def is_deepseek_key(key: str) -> bool:
    """True when `key` selects the DeepSeek backend, bare or model-specific.

    The colon is what makes this extensible: ``deepseek`` alone is the configured
    default, while anything of the form ``deepseek:<id>`` pins one model.
    """
    return bool(key) and (key == DEEPSEEK_KEY or key.startswith(DEEPSEEK_KEY_PREFIX))


def deepseek_model_for(key: str, default: str) -> str:
    """Resolve a DeepSeek build-model key to the model id sent upstream.

    ``deepseek`` → `default` (DEEPSEEK_MODEL); ``deepseek:<id>`` → ``<id>``. A
    key of ``deepseek:`` with nothing after it is treated as the bare key rather
    than sent upstream as an empty model name.
    """
    if key.startswith(DEEPSEEK_KEY_PREFIX):
        return key[len(DEEPSEEK_KEY_PREFIX):].strip() or default
    return default


def deepseek_key_for(model_id: str) -> str:
    """The catalogue key that pins a specific DeepSeek model.

    The inverse of `deepseek_model_for`, and the single place the
    ``deepseek:<id>`` spelling is written — the admin portal builds catalogue
    entries with it when listing the provider's models.
    """
    return f"{DEEPSEEK_KEY_PREFIX}{model_id.strip()}"


class BuildModelDef(BaseModel):
    """A single build-model catalogue entry, rendered by UIs and validated by the API."""

    model: str
    label: str = ""
    provider: str = ""
    sub: str = ""


def coerce_build_model(entry: dict) -> BuildModelDef:
    """Normalise a Firestore model dict into a BuildModelDef, filling gaps.

    Tolerant of partial writes / legacy docs so the picker never blanks out.
    """
    key = str(entry.get("model", "")).strip()
    return BuildModelDef(
        model=key,
        label=entry.get("label") or key or "",
        provider=entry.get("provider") or "",
        sub=entry.get("sub") or "",
    )


# Default catalogue offered to users for selection, in display order. The
# ``model`` value is what the rest of the backend recognises; ``label``,
# ``provider`` and ``sub`` exist purely to render the selection UI and never
# influence which adapter runs a build. Each entry has its own dedicated key
# (see app/config.py) — none of these route through a third-party proxy.
DEFAULT_BUILD_MODELS: tuple[dict[str, str], ...] = (
    {"model": "gemini", "label": "Gemini", "provider": "Google", "sub": "fast & capable"},
    {"model": "claude", "label": "Claude", "provider": "Anthropic", "sub": "precise reasoning"},
    {"model": "gpt", "label": "GPT-4o", "provider": "OpenAI", "sub": ""},
    # DeepSeek's two live models, each on DeepSeek's own key (api.deepseek.com).
    # `deepseek` tracks DEEPSEEK_MODEL; the pinned key selects a model outright.
    # Both support tool calls + JSON, which the build agent requires. The bare key
    # is kept so a platform that pins DEEPSEEK_MODEL still has one obvious entry.
    {"model": "deepseek", "label": "DeepSeek", "provider": "DeepSeek",
     "sub": "direct API key · server default"},
    {"model": "deepseek:deepseek-flash", "label": "DeepSeek Flash",
     "provider": "DeepSeek", "sub": "V4.1 Flash · 1M ctx · vision"},
    {"model": "deepseek:deepseek-v4-pro", "label": "DeepSeek V4 Pro",
     "provider": "DeepSeek", "sub": "V4 Pro · 1M ctx · strongest"},
    {"model": "Qwen3", "label": "Qwen3", "provider": "Open models", "sub": "OpenRouter / Ollama"},
)


def _default_model_keys() -> frozenset[str]:
    return frozenset(m["model"] for m in DEFAULT_BUILD_MODELS)


async def get_available_build_models(db, settings) -> list[dict]:
    """Return the build models offered to users for selection.

    Reads the admin-curated list from Firestore ``system/config.build_models``;
    falls back to ``DEFAULT_BUILD_MODELS`` when the field is unset, empty, or
    malformed (e.g. a corrupted document mid-write). Errors are non-fatal — a
    Firestore blip must never empty the model picker.
    """
    try:
        doc = await db.collection("system").document("config").get()
        if doc.exists:
            configured = (doc.to_dict() or {}).get("build_models")
            if isinstance(configured, list) and configured:
                return configured  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001 — a probe must not crash the caller
        logger.warning(
            "Could not read build_models from Firestore: %s — falling back to defaults", exc
        )
    return [dict(m) for m in DEFAULT_BUILD_MODELS]


async def get_valid_build_model_keys(db, settings) -> frozenset[str]:
    """The set of model identifiers assignable as a build model.

    Derived from the admin-curated catalogue so models an operator adds from the
    dashboard are assignable here, while an unset/empty catalogue falls back to
    the shipped defaults.
    """
    models = await get_available_build_models(db, settings)
    keys = frozenset(
        m["model"] for m in models if isinstance(m, dict) and m.get("model")
    )
    return keys or _default_model_keys()


async def get_user_build_model(db, user_id: str) -> str | None:
    """Return the user's own build-model override, if they've set one."""
    if not user_id:
        return None
    try:
        doc = await db.collection("users").document(user_id).get()
        if doc.exists:
            return (doc.to_dict() or {}).get("build_model") or None
    except Exception as exc:
        logger.warning("Could not read user build model for user=%s: %s", user_id, exc)
    return None


async def get_effective_build_model(db, settings, user_id: str | None = None) -> str:
    """Return the build model to use: the user's own override, else the Firestore
    system/config override, else the .env default."""
    if user_id:
        user_model = await get_user_build_model(db, user_id)
        if user_model:
            return user_model

    try:
        doc = await db.collection("system").document("config").get()
        if doc.exists:
            model = (doc.to_dict() or {}).get("build_model")
            if model:
                return model
    except Exception as exc:
        logger.warning("Could not read build model from Firestore: %s — falling back to .env", exc)
    return settings.BUILD_MODEL
