"""Tests for EditWorkspace.ensure() precondition checks.

Run:
    venv/Scripts/python -m pytest tests/test_edit_workspace_ensure.py -v
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from app.build.workspace import EditWorkspace

PROJECT_ID = uuid.UUID("14cb68a9-5b7e-43da-a94f-fff10c7bca45")


class TestEnsureRejectsMissingRepo:
    def test_empty_repo_full_name_raises_before_any_git_call(self) -> None:
        ws = EditWorkspace(PROJECT_ID, "", "gh_token")
        with patch("app.build.workspace._run") as run:
            with pytest.raises(RuntimeError, match="never.*published to GitHub"):
                ws.ensure()
        run.assert_not_called()

    def test_message_names_the_real_cause_not_a_git_error(self) -> None:
        ws = EditWorkspace(PROJECT_ID, "", "gh_token")
        with patch("app.build.workspace._run"):
            with pytest.raises(RuntimeError) as excinfo:
                ws.ensure()
        # No token-bearing garbage URL in the message the operator sees.
        assert "github.com/.git" not in str(excinfo.value)
        assert "repo_full_name" in str(excinfo.value)
