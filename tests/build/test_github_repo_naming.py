"""Repo-name collision handling.

Run:
    venv/Scripts/python -m pytest tests/build/test_github_repo_naming.py -v

Building an app whose name is already on the account used to fail the whole
build with "rename your app and try again". A taken name is an ordinary
outcome, so the client suffixes it instead.
"""
from __future__ import annotations

import pytest

from app.build import github_client as gc
from app.build.github_client import GitHubClient

_TAKEN = "GitHub create_repo failed 422: name already exists on this account"


class _FakeGitHub(GitHubClient):
    """Client whose create_repo rejects a fixed set of names as taken."""

    def __init__(self, taken: set[str], fail_with: str = _TAKEN):
        super().__init__("token")
        self.taken = taken
        self.fail_with = fail_with
        self.attempts: list[str] = []

    def create_repo(self, name, description="", private=True) -> dict:
        self.attempts.append(name)
        if name in self.taken:
            raise RuntimeError(self.fail_with)
        return {"name": name, "full_name": f"user/{name}", "html_url": f"https://x/{name}"}


class TestSuffixing:
    def test_free_name_is_used_as_is(self):
        gh = _FakeGitHub(taken=set())
        assert gh.create_repo_unique("todo")["name"] == "todo"
        assert gh.attempts == ["todo"]

    def test_taken_name_gets_a_numeric_suffix(self):
        gh = _FakeGitHub(taken={"todo"})
        assert gh.create_repo_unique("todo")["name"] == "todo-2"
        assert gh.attempts == ["todo", "todo-2"]

    def test_walks_the_sequence_until_one_is_free(self):
        gh = _FakeGitHub(taken={"todo", "todo-2", "todo-3"})
        assert gh.create_repo_unique("todo")["name"] == "todo-4"

    def test_falls_back_to_a_random_suffix_past_the_numeric_range(self):
        taken = {"todo"} | {f"todo-{n}" for n in range(2, gc._MAX_NUMERIC_SUFFIX + 1)}
        gh = _FakeGitHub(taken=taken)
        name = gh.create_repo_unique("todo")["name"]
        assert name.startswith("todo-")
        assert name not in taken
        # Random tail, not another integer in the exhausted sequence.
        assert not name.rsplit("-", 1)[1].isdigit() or len(name.rsplit("-", 1)[1]) == 4

    def test_returns_full_name_for_the_chosen_repo(self):
        gh = _FakeGitHub(taken={"todo"})
        assert gh.create_repo_unique("todo")["full_name"] == "user/todo-2"


class TestFailuresThatAreNotCollisions:
    def test_other_errors_propagate_untouched(self):
        """Auth/quota failures must not be retried under a different name."""
        gh = _FakeGitHub(taken={"todo"}, fail_with="GitHub create_repo failed 401: Bad credentials")
        with pytest.raises(RuntimeError, match="Bad credentials"):
            gh.create_repo_unique("todo")
        assert gh.attempts == ["todo"], "a non-collision error must not be retried"

    def test_gives_up_after_max_attempts(self):
        class AlwaysTaken(_FakeGitHub):
            def create_repo(self, name, description="", private=True) -> dict:
                self.attempts.append(name)
                raise RuntimeError(_TAKEN)

        gh = AlwaysTaken(taken=set())
        with pytest.raises(RuntimeError, match="Could not find a free"):
            gh.create_repo_unique("todo", max_attempts=4)
        assert len(gh.attempts) == 4


class TestNameConstraints:
    def test_detects_only_genuine_collision_messages(self):
        assert gc._is_name_taken("422: name already exists on this account")
        assert not gc._is_name_taken("401: Bad credentials")
        assert not gc._is_name_taken("422: name is invalid")

    def test_suffixed_name_respects_githubs_length_cap(self):
        long_base = "a" * 140
        for attempt in (1, 2, gc._MAX_NUMERIC_SUFFIX + 1):
            assert len(gc._suffixed_name(long_base, attempt)) <= gc._MAX_NAME_LEN

    def test_long_name_keeps_its_suffix_intact(self):
        """Truncation must eat the base, never the disambiguating suffix."""
        assert gc._suffixed_name("b" * 140, 2).endswith("-2")
