"""Tests for the deterministic launch-page demo guard.

Run:
    venv/Scripts/python -m pytest tests/build/test_demo_guard.py --confcutdir=tests/build -v
"""
from __future__ import annotations

from pathlib import Path

from app.build.demo_guard import _demo_evidence, _launch_page, demo_screen_present


def _write(path: Path, rel: str, text: str) -> None:
    target = path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


class TestLaunchPageDiscovery:
    def test_expo_index_is_found(self, tmp_path):
        _write(tmp_path, "app/index.tsx", "export default function Home() { return null; }")
        assert _launch_page(tmp_path, "react_native") == tmp_path / "app" / "index.tsx"

    def test_next_app_router_page_is_found(self, tmp_path):
        _write(tmp_path, "app/page.tsx", "export default function Page() { return null; }")
        assert _launch_page(tmp_path, "next") == tmp_path / "app" / "page.tsx"

    def test_flutter_resolves_main_dart_home_widget(self, tmp_path):
        _write(tmp_path, "lib/main.dart",
               "void main() => runApp(App());\nclass App extends StatelessWidget {\n  Widget build(_) => HomeScreen();\n  final home = HomeWidget();\n}")
        # main.dart has no `home:` widget here — guard must not crash or guess.
        assert demo_screen_present(tmp_path, "flutter") == (False, "")

    def test_unknown_template_returns_none(self, tmp_path):
        assert _launch_page(tmp_path, "vanilla") is None


class TestDemoDetection:
    def test_strong_marker_alone_flags(self):
        hit = _demo_evidence("Welcome to your app — Lorem ipsum dolor sit amet.")
        assert any("lorem ipsum" in e for e in hit)

    def test_single_weak_marker_does_not_flag(self):
        # A real app may legitimately say 'Welcome to <AppName>'.
        assert _demo_evidence("Welcome to TaskFlow — organise your day.") == []

    def test_two_weak_markers_flag(self):
        assert _demo_evidence("A demo template with sample data.") != []

    def test_real_home_content_passes(self, tmp_path):
        _write(tmp_path, "app/index.tsx",
               "export default function Home() {\n  return <TaskList tasks={tasks} />;\n}")
        assert demo_screen_present(tmp_path, "react_native") == (False, "")

    def test_demo_home_is_flagged(self, tmp_path):
        _write(tmp_path, "app/index.tsx",
               "export default function Demo() {\n  return <Text>Lorem ipsum — starter template</Text>;\n}")
        hit, evidence = demo_screen_present(tmp_path, "react_native")
        assert hit and "index.tsx" in evidence

    def test_never_raises_on_garbage_workspace(self, tmp_path):
        (tmp_path / "package.json").write_text("{broken", encoding="utf-8")
        assert demo_screen_present(tmp_path, "next")[0] is False
