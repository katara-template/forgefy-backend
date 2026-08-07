"""Tests for local-vs-cloud resolution in app/ai/ollama_http.py.

Run:
    venv/Scripts/python -m pytest tests/test_ollama_http.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai import ollama_http


def S(url: str = "", key: str = "", model: str = "", build_model: str = ""):
    return SimpleNamespace(
        OLLAMA_URL=url, OLLAMA_API_KEY=key,
        OLLAMA_MODEL=model, OLLAMA_BUILD_MODEL=build_model,
    )


class TestBaseUrlResolution:
    @pytest.mark.parametrize("url", [
        "", "http://ollama:11434", "http://localhost:11434", "http://127.0.0.1:11434",
    ])
    def test_local_urls_are_replaced_by_cloud_when_keyed(self, url):
        """A key with a Docker-internal host can never work — cloud wins."""
        assert ollama_http.ollama_base_url(S(url, "k")) == ollama_http.CLOUD_URL

    @pytest.mark.parametrize("url", ["", "http://ollama:11434"])
    def test_local_urls_are_kept_without_a_key(self, url):
        assert ollama_http.ollama_base_url(S(url, "")) == "http://ollama:11434"

    def test_explicit_host_is_respected_even_with_a_key(self):
        """Self-hosted endpoints and proxies must not be hijacked."""
        assert ollama_http.ollama_base_url(S("https://gpu.internal:11434", "k")) == \
            "https://gpu.internal:11434"

    def test_trailing_slash_is_stripped(self):
        assert ollama_http.ollama_base_url(S("https://gpu.internal/", "k")) == \
            "https://gpu.internal"


class TestHeaders:
    def test_no_auth_header_without_a_key(self):
        assert ollama_http.ollama_headers(S("x", "")) == {}

    def test_bearer_header_with_a_key(self):
        assert ollama_http.ollama_headers(S("x", "secret")) == \
            {"Authorization": "Bearer secret"}

    def test_whitespace_only_key_is_not_a_key(self):
        assert ollama_http.ollama_headers(S("x", "   ")) == {}
        assert not ollama_http.using_cloud(S("x", "   "))


class TestOptions:
    def test_local_run_caps_context_to_avoid_oom(self, monkeypatch):
        monkeypatch.setattr(ollama_http, "using_cloud", lambda s=None: False)
        assert ollama_http.ollama_options(8192, 4096) == \
            {"num_ctx": 8192, "num_predict": 4096}

    def test_cloud_run_keeps_its_full_window(self, monkeypatch):
        """Capping num_ctx on a 256K model would truncate long transcripts."""
        monkeypatch.setattr(ollama_http, "using_cloud", lambda s=None: True)
        assert ollama_http.ollama_options(8192, 4096) == {"num_predict": 4096}


class TestBuildModelSplit:
    def test_falls_back_to_the_general_model(self):
        assert ollama_http.ollama_build_model(S(model="nemotron-3-super")) == \
            "nemotron-3-super"

    def test_build_model_overrides_when_set(self):
        assert ollama_http.ollama_build_model(
            S(model="nemotron-3-super", build_model="minimax-m3")
        ) == "minimax-m3"

    def test_blank_build_model_is_ignored(self):
        assert ollama_http.ollama_build_model(S(model="a", build_model="   ")) == "a"


class TestErrorHints:
    def test_missing_model_hint_names_the_model(self):
        assert "nemotron-3-super" in ollama_http.missing_model_hint("nemotron-3-super")

    def test_auth_hint_only_fires_on_auth_codes(self):
        assert ollama_http.auth_error_hint(200) is None
        assert ollama_http.auth_error_hint(500) is None
        assert ollama_http.auth_error_hint(401)
        assert ollama_http.auth_error_hint(403)
