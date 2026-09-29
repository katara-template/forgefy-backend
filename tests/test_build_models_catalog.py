"""Tests for the admin-curated build-model catalogue.

These exercise the catalogue resolution shared by ``app.core.build_model``
(fallback → custom → defaults → non-fatal errors) without booting the server.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.build_model import (
    DEFAULT_BUILD_MODELS,
    VALID_BUILD_MODELS,
    coerce_build_model,
    deepseek_model_for,
    get_available_build_models,
    get_valid_build_model_keys,
    is_deepseek_key,
)


def _doc(exists: bool, data: dict) -> MagicMock:
    m = MagicMock()
    m.exists = exists
    m.to_dict.return_value = data
    return m


def _db_returning(doc: MagicMock) -> MagicMock:
    """A fake async-firestore db whose system/config get() resolves to ``doc``."""
    db = MagicMock()
    db.collection.return_value.document.return_value.get = AsyncMock(return_value=doc)
    return db


def _default_keys() -> frozenset[str]:
    return frozenset(m["model"] for m in DEFAULT_BUILD_MODELS)


@pytest.mark.asyncio
async def test_custom_catalogue_used_for_keys() -> None:
    custom = [dict(m) for m in DEFAULT_BUILD_MODELS[:2]]
    db = _db_returning(_doc(True, {"build_models": custom}))
    keys = await get_valid_build_model_keys(db, None)
    assert keys == {"gemini", "claude"}


@pytest.mark.asyncio
async def test_custom_catalogue_returned_by_available_models() -> None:
    custom = [dict(m) for m in DEFAULT_BUILD_MODELS[:2]]
    db = _db_returning(_doc(True, {"build_models": custom}))
    models = await get_available_build_models(db, None)
    assert len(models) == 2
    assert {coerce_build_model(m).model for m in models} == {"gemini", "claude"}


@pytest.mark.asyncio
async def test_missing_doc_falls_back_to_defaults() -> None:
    db = _db_returning(_doc(False, {}))
    keys = await get_valid_build_model_keys(db, None)
    assert keys == _default_keys()


@pytest.mark.asyncio
async def test_empty_list_falls_back_to_defaults() -> None:
    db = _db_returning(_doc(True, {"build_models": []}))
    keys = await get_valid_build_model_keys(db, None)
    assert keys == _default_keys()


@pytest.mark.asyncio
async def test_malformed_entry_is_ignored_safely() -> None:
    """A catalogue containing a non-dict (corrupt write) is skipped, not crashed."""
    db = _db_returning(_doc(True, {"build_models": ["not-a-dict", {"model": "gemini"}]}))
    keys = await get_valid_build_model_keys(db, None)
    assert keys == {"gemini"}


@pytest.mark.asyncio
async def test_firestore_error_falls_back_to_defaults() -> None:
    db = MagicMock()
    db.collection.side_effect = RuntimeError("boom")
    keys = await get_valid_build_model_keys(db, None)
    assert keys == _default_keys()


def test_valid_build_models_tuple_unchanged() -> None:
    # The provider-level keys the worker ships adapters for; the catalogue is
    # validated by shape. deepseek joined as its own direct-API provider (not a
    # proxy), and `deepseek:<model-id>` pins one model within it.
    assert set(VALID_BUILD_MODELS) == {"claude", "Qwen3", "gemini", "gpt", "deepseek"}


def test_deepseek_key_helpers() -> None:
    assert is_deepseek_key("deepseek") is True
    assert is_deepseek_key("deepseek:deepseek-v4-pro") is True
    # A near-miss is not a DeepSeek key — only the colon form extends it.
    assert is_deepseek_key("deepseek-v4-pro") is False
    assert is_deepseek_key("") is False

    # Bare key tracks the configured default; the suffix pins a model outright.
    assert deepseek_model_for("deepseek", "deepseek-flash") == "deepseek-flash"
    assert deepseek_model_for("deepseek:deepseek-v4-pro", "deepseek-flash") == "deepseek-v4-pro"
    # A dangling colon must not send an empty model name upstream.
    assert deepseek_model_for("deepseek:", "deepseek-flash") == "deepseek-flash"


def test_coerce_build_model_fills_gaps() -> None:
    c = coerce_build_model({"model": "x", "label": "", "provider": "p"})
    assert c.model == "x"
    assert c.label == "x"  # empty label falls back to the model key
    assert c.provider == "p"
    assert c.sub == ""


def test_default_catalogue_is_complete() -> None:
    """Every default catalogue entry must resolve to a provider the worker can drive.

    Checked by provider rather than by exact key because DeepSeek entries may
    carry a ``deepseek:<model-id>`` suffix to pin a specific model.
    """
    for entry in DEFAULT_BUILD_MODELS:
        provider = entry["model"].split(":", 1)[0]
        assert provider in VALID_BUILD_MODELS, entry


def test_default_catalogue_offers_deepseek_as_a_direct_provider() -> None:
    """DeepSeek ships in the default catalogue with its own vendor identity.

    It must not be described as an OpenRouter/Ollama route — a user picking it is
    choosing DeepSeek's own API key against api.deepseek.com.
    """
    entry = next(m for m in DEFAULT_BUILD_MODELS if m["model"] == "deepseek")
    assert entry["label"] == "DeepSeek"
    assert entry["provider"] == "DeepSeek"
    assert "openrouter" not in entry["sub"].lower()
    assert "ollama" not in entry["sub"].lower()
