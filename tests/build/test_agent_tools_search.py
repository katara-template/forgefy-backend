"""Tests for the search and partial-edit tools: edit_file, grep, glob, read_file.

Run:
    venv/Scripts/python -m pytest tests/build/test_agent_tools_search.py -v

grep is exercised twice where it matters — once through ripgrep if the binary is
present, and once through the pure-Python fallback — because the fallback is what
runs on any worker image without rg installed.
"""
from __future__ import annotations

import shutil
from unittest.mock import patch

import pytest

from app.build.agent_tools import TOOLS, execute_tool

_HAS_RG = bool(shutil.which("rg") or shutil.which("rg.exe"))


@pytest.fixture
def project(tmp_path):
    """A small workspace with a bit of realistic structure."""
    (tmp_path / "src" / "lib").mkdir(parents=True)
    (tmp_path / "node_modules" / "junk").mkdir(parents=True)
    (tmp_path / "src" / "app.ts").write_text(
        "import { greet } from './lib/greet';\n"
        "export function main() {\n"
        "  console.log(greet('world'));\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "lib" / "greet.ts").write_text(
        "export function greet(name: string): string {\n"
        "  return `Hello, ${name}`;\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Demo\n\ngreet the world\n", encoding="utf-8")
    (tmp_path / "node_modules" / "junk" / "index.js").write_text(
        "function greet() { return 'vendored'; }\n", encoding="utf-8",
    )
    return tmp_path


def _run(name, inputs, workspace):
    return execute_tool(name, inputs, workspace)


# ── registration ──────────────────────────────────────────────────────────────


class TestToolRegistration:
    @pytest.mark.parametrize("name", ["edit_file", "grep", "glob"])
    def test_tool_is_registered(self, name):
        tool = next((t for t in TOOLS if t["name"] == name), None)

        assert tool is not None, f"{name} is not exposed to the model"
        assert tool["input_schema"]["type"] == "object"
        assert tool["description"].strip()

    def test_read_file_advertises_offset_and_limit(self):
        tool = next(t for t in TOOLS if t["name"] == "read_file")
        props = tool["input_schema"]["properties"]

        assert "offset" in props and "limit" in props
        assert tool["input_schema"]["required"] == ["path"]

    def test_write_file_description_points_at_edit_file(self):
        tool = next(t for t in TOOLS if t["name"] == "write_file")

        assert "edit_file" in tool["description"]


# ── edit_file ─────────────────────────────────────────────────────────────────


class TestEditFile:
    def test_replaces_a_unique_string_and_returns_a_diff(self, project):
        result = _run(
            "edit_file",
            {"path": "src/lib/greet.ts", "old_string": "Hello, ", "new_string": "Hi, "},
            project,
        )

        assert result.startswith("OK:")
        assert "Hi, ${name}" in (project / "src" / "lib" / "greet.ts").read_text()
        assert "-  return `Hello, ${name}`;" in result
        assert "+  return `Hi, ${name}`;" in result

    def test_ambiguous_match_is_refused_with_actionable_advice(self, project):
        (project / "dup.ts").write_text("const a = 1;\nconst a = 1;\n", encoding="utf-8")

        result = _run(
            "edit_file",
            {"path": "dup.ts", "old_string": "const a = 1;", "new_string": "const a = 2;"},
            project,
        )

        assert "ERROR" in result
        assert "appears 2 times" in result
        assert "replace_all" in result
        # The file must be untouched when the edit is refused.
        assert (project / "dup.ts").read_text() == "const a = 1;\nconst a = 1;\n"

    def test_replace_all_changes_every_occurrence(self, project):
        (project / "dup.ts").write_text("const a = 1;\nconst a = 1;\n", encoding="utf-8")

        result = _run(
            "edit_file",
            {
                "path": "dup.ts",
                "old_string": "const a = 1;",
                "new_string": "const a = 2;",
                "replace_all": True,
            },
            project,
        )

        assert "replaced 2 occurrence(s)" in result
        assert (project / "dup.ts").read_text() == "const a = 2;\nconst a = 2;\n"

    def test_missing_string_is_reported_without_writing(self, project):
        before = (project / "src" / "app.ts").read_text()

        result = _run(
            "edit_file",
            {"path": "src/app.ts", "old_string": "not in the file", "new_string": "x"},
            project,
        )

        assert "ERROR" in result and "not found" in result
        assert (project / "src" / "app.ts").read_text() == before

    def test_editing_a_missing_file_points_at_write_file(self, project):
        result = _run(
            "edit_file", {"path": "nope.ts", "old_string": "a", "new_string": "b"}, project,
        )

        assert "ERROR" in result
        assert "write_file" in result

    def test_no_op_edit_is_refused(self, project):
        result = _run(
            "edit_file",
            {"path": "src/app.ts", "old_string": "main", "new_string": "main"},
            project,
        )

        assert "ERROR" in result and "identical" in result

    def test_deleting_text_with_an_empty_replacement(self, project):
        result = _run(
            "edit_file",
            {"path": "README.md", "old_string": "greet the world\n", "new_string": ""},
            project,
        )

        assert result.startswith("OK:")
        assert "greet the world" not in (project / "README.md").read_text()

    def test_cannot_edit_outside_the_workspace(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()
        outsider = tmp_path / "outside.txt"
        outsider.write_text("secret", encoding="utf-8")

        result = _run(
            "edit_file",
            {"path": "../outside.txt", "old_string": "secret", "new_string": "hacked"},
            workspace,
        )

        assert "ERROR" in result
        assert outsider.read_text() == "secret"


# ── read_file ─────────────────────────────────────────────────────────────────


class TestReadFile:
    def test_lines_are_numbered(self, project):
        result = _run("read_file", {"path": "src/lib/greet.ts"}, project)

        assert result.splitlines()[0].startswith("1\t")
        assert "export function greet" in result

    def test_offset_and_limit_window_the_file(self, project):
        (project / "many.txt").write_text(
            "\n".join(f"line {i}" for i in range(1, 101)), encoding="utf-8",
        )

        result = _run("read_file", {"path": "many.txt", "offset": 10, "limit": 3}, project)

        assert "[lines 10-12 of 100]" in result
        assert "line 10" in result and "line 12" in result
        assert "line 13" not in result
        assert "offset=13" in result, "the model needs to know how to continue"

    def test_default_read_is_capped_and_says_how_to_continue(self, project):
        (project / "huge.txt").write_text(
            "\n".join(f"line {i}" for i in range(1, 3001)), encoding="utf-8",
        )

        result = _run("read_file", {"path": "huge.txt"}, project)

        assert "[lines 1-2000 of 3000]" in result
        assert "offset=2001" in result

    def test_offset_past_the_end_is_an_error_not_a_crash(self, project):
        result = _run("read_file", {"path": "README.md", "offset": 9999}, project)

        assert "ERROR" in result

    def test_empty_file_is_reported(self, project):
        (project / "empty.txt").write_text("", encoding="utf-8")

        assert "empty" in _run("read_file", {"path": "empty.txt"}, project)

    def test_reading_a_directory_points_at_list_files(self, project):
        result = _run("read_file", {"path": "src"}, project)

        assert "ERROR" in result and "list_files" in result

    def test_path_only_call_still_works(self, project):
        """Existing callers pass nothing but a path."""
        assert "greet" in _run("read_file", {"path": "src/app.ts"}, project)


# ── grep ──────────────────────────────────────────────────────────────────────


class TestGrep:
    def test_finds_matching_lines_with_file_and_line_number(self, project):
        result = _run("grep", {"pattern": "function greet"}, project)

        assert "greet.ts" in result
        assert "1" in result

    def test_vendored_directories_are_skipped(self, project):
        result = _run("grep", {"pattern": "greet"}, project)

        assert "node_modules" not in result, "vendored code drowns out the project"

    def test_files_with_matches_mode_lists_paths_only(self, project):
        result = _run(
            "grep", {"pattern": "greet", "output_mode": "files_with_matches"}, project,
        )

        assert "greet.ts" in result
        assert "return `Hello" not in result

    def test_count_mode_returns_counts(self, project):
        result = _run("grep", {"pattern": "greet", "output_mode": "count"}, project)

        assert any(":" in line for line in result.splitlines())

    def test_case_insensitive_flag(self, project):
        assert "No matches" in _run("grep", {"pattern": "GREET"}, project)
        assert "No matches" not in _run(
            "grep", {"pattern": "GREET", "case_insensitive": True}, project,
        )

    def test_glob_filter_narrows_the_search(self, project):
        result = _run("grep", {"pattern": "greet", "glob": "*.md"}, project)

        assert "README.md" in result
        assert "greet.ts" not in result

    def test_no_match_is_a_clear_message_not_an_error(self, project):
        result = _run("grep", {"pattern": "zzzznotpresent"}, project)

        assert "No matches" in result
        assert "ERROR" not in result

    def test_head_limit_truncates_with_a_hint(self, project):
        (project / "big.txt").write_text(
            "\n".join("match me" for _ in range(50)), encoding="utf-8",
        )

        result = _run("grep", {"pattern": "match me", "head_limit": 5}, project)

        assert len([ln for ln in result.splitlines() if "match me" in ln]) == 5
        assert "more result lines" in result

    def test_search_is_confined_to_the_workspace(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()
        (workspace / "a.txt").write_text("inside", encoding="utf-8")
        (tmp_path / "outside.txt").write_text("secret-token", encoding="utf-8")

        result = _run("grep", {"pattern": "secret-token", "path": ".."}, workspace)

        assert "secret-token" not in result
        assert "ERROR" in result

    def test_invalid_regex_is_reported_by_the_fallback(self, project):
        with patch("app.build.agent_tools.shutil.which", return_value=None):
            result = _run("grep", {"pattern": "([unclosed"}, project)

        assert "ERROR" in result

    # The fallback is what runs on an image without ripgrep, so it gets the same
    # behavioural assertions as the rg path.
    def test_fallback_finds_matching_lines(self, project):
        with patch("app.build.agent_tools.shutil.which", return_value=None):
            result = _run("grep", {"pattern": "function greet"}, project)

        assert "greet.ts" in result

    def test_fallback_skips_vendored_directories(self, project):
        with patch("app.build.agent_tools.shutil.which", return_value=None):
            result = _run("grep", {"pattern": "greet"}, project)

        assert "node_modules" not in result

    def test_fallback_files_with_matches_mode(self, project):
        with patch("app.build.agent_tools.shutil.which", return_value=None):
            result = _run(
                "grep", {"pattern": "greet", "output_mode": "files_with_matches"}, project,
            )

        assert "greet.ts" in result
        assert "return `Hello" not in result

    def test_fallback_context_lines(self, project):
        with patch("app.build.agent_tools.shutil.which", return_value=None):
            result = _run(
                "grep", {"pattern": "return `Hello", "context_lines": 1}, project,
            )

        assert "export function greet" in result

    @pytest.mark.skipif(not _HAS_RG, reason="ripgrep not installed")
    def test_ripgrep_is_invoked_without_a_shell(self, project):
        import subprocess as sp

        real = sp.run
        seen = {}

        def spy(argv, **kwargs):
            seen["argv"] = argv
            seen["shell"] = kwargs.get("shell", False)
            return real(argv, **kwargs)

        with patch("app.build.agent_tools.subprocess.run", side_effect=spy):
            _run("grep", {"pattern": "greet"}, project)

        assert isinstance(seen["argv"], list), "argv must be a list, never a shell string"
        assert seen["shell"] is False


# ── glob ──────────────────────────────────────────────────────────────────────


class TestGlob:
    def test_finds_files_by_pattern(self, project):
        result = _run("glob", {"pattern": "src/**/*.ts"}, project)

        assert "greet.ts" in result
        assert "app.ts" in result

    def test_non_matching_extension_is_excluded(self, project):
        result = _run("glob", {"pattern": "**/*.ts"}, project)

        assert "README.md" not in result

    def test_results_are_newest_first(self, project):
        import os
        import time

        old = project / "src" / "app.ts"
        new = project / "src" / "lib" / "greet.ts"
        now = time.time()
        os.utime(old, (now - 500, now - 500))
        os.utime(new, (now, now))

        lines = _run("glob", {"pattern": "src/**/*.ts"}, project).splitlines()

        assert "greet.ts" in lines[0], "the most recently touched file comes first"

    def test_no_match_is_a_clear_message(self, project):
        assert "No files match" in _run("glob", {"pattern": "**/*.rs"}, project)

    def test_vendored_directories_are_skipped(self, project):
        result = _run("glob", {"pattern": "**/*.js"}, project)

        assert "node_modules" not in result

    def test_path_scopes_the_search(self, project):
        result = _run("glob", {"pattern": "**/*.ts", "path": "src/lib"}, project)

        assert "greet.ts" in result
        assert "app.ts" not in result

    def test_cannot_glob_outside_the_workspace(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()
        (tmp_path / "outside.ts").write_text("x", encoding="utf-8")

        result = _run("glob", {"pattern": "*.ts", "path": ".."}, workspace)

        assert "ERROR" in result

    def test_result_count_is_capped(self, project):
        many = project / "many"
        many.mkdir()
        for i in range(250):
            (many / f"f{i}.ts").write_text("x", encoding="utf-8")

        result = _run("glob", {"pattern": "many/*.ts"}, project)

        assert "more matches not shown" in result
        assert len(result.splitlines()) <= 201
