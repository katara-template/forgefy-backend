"""Credential handling around git operations.

Run:
    venv/Scripts/python -m pytest tests/build/test_workspace_credentials.py -v

Two separate concerns, both learned from a production failure:

  * Git authenticates by embedding the token in the URL, so any error derived
    from the command leaks a working credential into logs and Sentry.
  * The build uses two different GitHub identities — the platform token for
    Forgefy's private template org, the user's token for the user's own repo.
    Swapping them fails ONLY for users who connected GitHub, which makes it look
    intermittent while being perfectly reproducible per user.
"""
from __future__ import annotations

import subprocess

import pytest

from app.build import workspace as ws


class TestRedaction:
    def test_token_in_url_is_removed(self):
        out = ws._redact("git clone https://gho_SECRETVALUE123@github.com/org/repo.git /tmp/x")
        assert "gho_SECRETVALUE123" not in out
        assert "https://***@github.com/org/repo.git" in out

    def test_basic_auth_pair_is_removed(self):
        out = ws._redact("https://user:hunter2@example.com/repo.git")
        assert "hunter2" not in out and "user" not in out

    def test_multiple_urls_all_redacted(self):
        out = ws._redact(
            "https://tok_AAA@github.com/a.git and https://tok_BBB@github.com/b.git"
        )
        assert "tok_AAA" not in out and "tok_BBB" not in out
        assert out.count("***") == 2

    def test_urls_without_credentials_are_untouched(self):
        url = "https://github.com/katara-template/next-ts.git"
        assert ws._redact(url) == url

    def test_non_url_text_is_untouched(self):
        msg = "fatal: repository not found"
        assert ws._redact(msg) == msg


class TestRunLeakage:
    def test_failed_command_does_not_leak_the_token(self, monkeypatch):
        """The exact production leak: a failed clone put a live token in Sentry."""
        token = "TOKEN HERE tt"
        url = f"https://{token}@github.com/katara-template/next-ts.git"

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a, returncode=128,
                stdout="", stderr=f"fatal: repository '{url}' not found",
            ),
        )

        with pytest.raises(RuntimeError) as exc:
            ws._run(["git", "clone", "--depth", "1", url, "/tmp/x"])

        message = str(exc.value)
        assert token not in message, "token leaked through the exception"
        assert "***" in message
        # Still has to be diagnosable.
        assert "not found" in message

    def test_successful_command_returns_stdout(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: subprocess.CompletedProcess(
                args=a, returncode=0, stdout="ok", stderr="",
            ),
        )
        assert ws._run(["git", "status"]) == "ok"


class TestTemplateCloneIdentity:
    def test_template_clone_uses_the_platform_token(self):
        """Cloning Forgefy's private template must not use a user's token.

        A user token 404s on katara-template/* — GitHub hides private repos
        rather than returning 403 — so builds break for exactly those users who
        connected their own GitHub account.
        """
        import inspect

        from app.workers import build_worker

        src = inspect.getsource(build_worker)
        start = src.index("workspace = Workspace(")
        call = src[start : src.index(")", src.index("git_token", start))]
        assert "settings.GITHUB_TOKEN" in call, (
            "template clone must use the platform token, not the resolved user token"
        )
        assert "git_token=github_token" not in call
