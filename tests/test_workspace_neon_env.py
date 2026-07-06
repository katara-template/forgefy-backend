"""Tests for Workspace/EditWorkspace's Neon Data API .env injection."""
from pathlib import Path

from app.build.workspace import NEON_ENV_VAR_NAMES, inject_neon_env


class TestInjectNeonEnv:
    def test_replaces_existing_placeholder_value(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            'NEXT_PUBLIC_NEON_DATA_API_URL="https://your-project.dataapi.neon.tech"\n'
            'NEXT_PUBLIC_OTHER_VAR="keep-me"\n',
            encoding="utf-8",
        )

        inject_neon_env(tmp_path, "next", "https://real.dataapi.neon.tech")

        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert 'NEXT_PUBLIC_NEON_DATA_API_URL="https://real.dataapi.neon.tech"' in content
        assert 'NEXT_PUBLIC_OTHER_VAR="keep-me"' in content
        assert content.count("NEXT_PUBLIC_NEON_DATA_API_URL") == 1

    def test_appends_when_env_file_missing(self, tmp_path: Path) -> None:
        inject_neon_env(tmp_path, "flutter", "https://real.dataapi.neon.tech")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert 'NEON_DATA_API_URL="https://real.dataapi.neon.tech"' in content

    def test_uses_framework_specific_variable_names(self, tmp_path: Path) -> None:
        inject_neon_env(tmp_path, "react_native", "https://x.dataapi.neon.tech")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "EXPO_PUBLIC_NEON_DATA_API_URL=" in content

    def test_unknown_template_key_falls_back_to_next(self, tmp_path: Path) -> None:
        inject_neon_env(tmp_path, "something-unrecognized", "https://x.dataapi.neon.tech")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "NEXT_PUBLIC_NEON_DATA_API_URL=" in content

    def test_var_names_table_covers_all_three_templates(self) -> None:
        assert set(NEON_ENV_VAR_NAMES) == {"next", "react_native", "flutter"}

    def test_coexists_with_supabase_values_in_same_file(self, tmp_path: Path) -> None:
        from app.build.workspace import inject_supabase_env

        inject_supabase_env(tmp_path, "next", "https://x.supabase.co", "anon-key")
        inject_neon_env(tmp_path, "next", "https://x.dataapi.neon.tech")

        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "NEXT_PUBLIC_SUPABASE_URL=" in content
        assert "NEXT_PUBLIC_SUPABASE_ANON_KEY=" in content
        assert "NEXT_PUBLIC_NEON_DATA_API_URL=" in content
