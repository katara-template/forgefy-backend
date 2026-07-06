"""Tests for Workspace/EditWorkspace's Firebase client-config .env injection."""
from pathlib import Path

from app.build.workspace import FIREBASE_ENV_VAR_NAMES, inject_firebase_env

_CONFIG = {
    "apiKey": "real-api-key",
    "authDomain": "real-project.firebaseapp.com",
    "projectId": "real-project",
    "storageBucket": "real-project.appspot.com",
    "messagingSenderId": "111222333",
    "appId": "1:111222333:web:deadbeef",
}


class TestInjectFirebaseEnv:
    def test_replaces_existing_placeholder_values(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            'NEXT_PUBLIC_FIREBASE_API_KEY="your-firebase-api-key"\n'
            'NEXT_PUBLIC_FIREBASE_PROJECT_ID="your-project-id"\n'
            'NEXT_PUBLIC_OTHER_VAR="keep-me"\n',
            encoding="utf-8",
        )

        inject_firebase_env(tmp_path, "next", _CONFIG)

        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert 'NEXT_PUBLIC_FIREBASE_API_KEY="real-api-key"' in content
        assert 'NEXT_PUBLIC_FIREBASE_PROJECT_ID="real-project"' in content
        assert 'NEXT_PUBLIC_OTHER_VAR="keep-me"' in content
        assert content.count("NEXT_PUBLIC_FIREBASE_API_KEY") == 1

    def test_appends_when_env_file_missing(self, tmp_path: Path) -> None:
        inject_firebase_env(tmp_path, "flutter", _CONFIG)
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert 'FIREBASE_API_KEY="real-api-key"' in content
        assert 'FIREBASE_PROJECT_ID="real-project"' in content
        assert 'FIREBASE_APP_ID="1:111222333:web:deadbeef"' in content

    def test_uses_framework_specific_variable_names(self, tmp_path: Path) -> None:
        inject_firebase_env(tmp_path, "react_native", _CONFIG)
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "EXPO_PUBLIC_FIREBASE_API_KEY=" in content
        assert "EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN=" in content
        assert "EXPO_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=" in content

    def test_unknown_template_key_falls_back_to_next(self, tmp_path: Path) -> None:
        inject_firebase_env(tmp_path, "something-unrecognized", _CONFIG)
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "NEXT_PUBLIC_FIREBASE_API_KEY=" in content

    def test_var_names_table_covers_all_three_templates(self) -> None:
        assert set(FIREBASE_ENV_VAR_NAMES) == {"next", "react_native", "flutter"}
        for names in FIREBASE_ENV_VAR_NAMES.values():
            assert set(names) == {
                "apiKey", "authDomain", "projectId",
                "storageBucket", "messagingSenderId", "appId",
            }

    def test_missing_config_keys_are_skipped_not_written_as_none(self, tmp_path: Path) -> None:
        partial = {"apiKey": "only-this-one"}
        inject_firebase_env(tmp_path, "next", partial)
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert 'NEXT_PUBLIC_FIREBASE_API_KEY="only-this-one"' in content
        assert "NEXT_PUBLIC_FIREBASE_PROJECT_ID" not in content

    def test_coexists_with_supabase_and_neon_values_in_same_file(self, tmp_path: Path) -> None:
        from app.build.workspace import inject_neon_env, inject_supabase_env

        inject_supabase_env(tmp_path, "next", "https://x.supabase.co", "anon-key")
        inject_neon_env(tmp_path, "next", "https://x.dataapi.neon.tech")
        inject_firebase_env(tmp_path, "next", _CONFIG)

        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "NEXT_PUBLIC_SUPABASE_URL=" in content
        assert "NEXT_PUBLIC_NEON_DATA_API_URL=" in content
        assert "NEXT_PUBLIC_FIREBASE_API_KEY=" in content
