"""Tests for Workspace/EditWorkspace's Supabase .env injection."""
from pathlib import Path

from app.build.workspace import SUPABASE_ENV_VAR_NAMES, inject_supabase_env


class TestInjectSupabaseEnv:
    def test_replaces_existing_placeholder_values(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            'NEXT_PUBLIC_SUPABASE_URL="your-project.supabase.co"\n'
            'NEXT_PUBLIC_SUPABASE_ANON_KEY="your-anon-key"\n'
            'NEXT_PUBLIC_OTHER_VAR="keep-me"\n',
            encoding="utf-8",
        )

        inject_supabase_env(tmp_path, "next", "https://real.supabase.co", "real-anon-key")

        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert 'NEXT_PUBLIC_SUPABASE_URL="https://real.supabase.co"' in content
        assert 'NEXT_PUBLIC_SUPABASE_ANON_KEY="real-anon-key"' in content
        assert 'NEXT_PUBLIC_OTHER_VAR="keep-me"' in content
        assert content.count("NEXT_PUBLIC_SUPABASE_URL") == 1

    def test_appends_when_env_file_missing(self, tmp_path: Path) -> None:
        inject_supabase_env(tmp_path, "flutter", "https://real.supabase.co", "real-anon-key")

        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert 'SUPABASE_URL="https://real.supabase.co"' in content
        assert 'SUPABASE_ANON_KEY="real-anon-key"' in content

    def test_uses_framework_specific_variable_names(self, tmp_path: Path) -> None:
        inject_supabase_env(tmp_path, "react_native", "https://x.supabase.co", "key")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "EXPO_PUBLIC_SUPABASE_URL=" in content
        assert "EXPO_PUBLIC_SUPABASE_ANON_KEY=" in content

    def test_unknown_template_key_falls_back_to_next(self, tmp_path: Path) -> None:
        inject_supabase_env(tmp_path, "something-unrecognized", "https://x.supabase.co", "key")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "NEXT_PUBLIC_SUPABASE_URL=" in content

    def test_var_names_table_covers_all_three_templates(self) -> None:
        assert set(SUPABASE_ENV_VAR_NAMES) == {"next", "react_native", "flutter"}
