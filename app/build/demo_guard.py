"""Deterministic launch-page guard — the mechanical backstop for the build
prompt's DEMO SCREEN mandate.

Prompts ask the agent to replace the template's demo/placeholder launch page and
to wire features as it goes; the validator re-checks it. Both are LLM-driven and
can be ignored. This module is not: after the build agent finishes, the worker
scans the actual entry-screen file for template-demo markers and, if any are
found, forces a targeted fix pass before anything is pushed.

Best-effort by design — a guard that raises would break builds worse than a
surviving demo page does.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Candidate launch-page files per template, in priority order. These are the
# pages users SEE first — layouts (_layout.tsx / _app.tsx) are wrappers, not
# content, so they are deliberately absent.
_LAUNCH_PAGE_TARGETS: dict[str, tuple[str, ...]] = {
    "react_native": (
        "app/index.tsx", "app/index.js",
        "app/(tabs)/index.tsx", "app/(tabs)/index.js",
        "src/app/index.tsx",
        "App.tsx", "App.js", "App.jsx",  # classic RN entry
    ),
    "next": (
        "app/page.tsx", "app/page.js",
        "src/app/page.tsx", "src/app/page.js",
        "pages/index.tsx", "pages/index.js",
        "src/pages/index.tsx", "src/pages/index.js",
    ),
}

# Markers that on their own indicate template/demo scaffolding.
_STRONG_MARKERS: tuple[str, ...] = (
    "lorem ipsum",
    "starter template",
    "template by",
    "demo screen",
    "demo page",
    "sample app",
    "this is a template",
    "forgefy template",
)

# Weak markers count only when several appear together — a real app can
# legitimately contain one of these words in isolation.
_WEAK_MARKERS: tuple[str, ...] = (
    "demo", "template", "sample data", "placeholder", "welcome to",
)


def _find_flutter_home(path: Path) -> Path | None:
    """Resolve Flutter's launch widget: main.dart's `home:` class → its file."""
    main = None
    for pubspec in sorted(path.rglob("pubspec.yaml")):
        if "node_modules" in str(pubspec) or ".dart_tool" in str(pubspec):
            continue
        main = pubspec.parent / "lib" / "main.dart"
        if main.is_file():
            break
    if main is None:
        return None
    try:
        text = main.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"home\s*:\s*(?:const\s+)?([A-Z]\w*)\s*\(", text)
    if not match:
        return None
    widget = match.group(1)
    for candidate in sorted((main.parent).rglob("*.dart")):
        try:
            body = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(rf"class\s+{re.escape(widget)}\b", body):
            return candidate
    return None


def _launch_page(path: Path, template_key: str) -> Path | None:
    """Locate the file holding the app's launch page, or None if unidentifiable."""
    base = path
    # Subdir projects: resolve against the dir holding package.json when present.
    pkg_roots = [p.parent for p in sorted(path.rglob("package.json"))
                 if "node_modules" not in str(p)]
    candidates_bases = [base] + pkg_roots

    for rel in _LAUNCH_PAGE_TARGETS.get(template_key, ()):
        for b in candidates_bases:
            candidate = b / rel
            if candidate.is_file():
                return candidate

    if template_key == "flutter":
        return _find_flutter_home(path)
    return None


def _demo_evidence(text: str) -> list[str]:
    """Return the marker lines that look like template-demo content."""
    lowered = text.lower()
    evidence: list[str] = []
    for marker in _STRONG_MARKERS:
        if marker in lowered:
            evidence.append(f"strong marker: {marker!r}")
    weak_hits = [m for m in _WEAK_MARKERS if m in lowered]
    if len(weak_hits) >= 2:
        evidence.append("weak markers x%d: %s" % (len(weak_hits), ", ".join(weak_hits)))
    return evidence


def demo_screen_present(path: Path, template_key: str) -> tuple[bool, str]:
    """Check whether the launch page still contains template-demo content.

    Returns (hit, evidence). Never raises: an unreadable or unidentifiable
    project simply reports False — the prompt mandate remains as first line of
    defense, and the validator re-checks regardless.
    """
    try:
        page = _launch_page(path, template_key)
        if page is None:
            return False, ""
        text = page.read_text(encoding="utf-8", errors="replace")
        evidence = _demo_evidence(text)
        if evidence:
            detail = "; ".join(evidence)
            logger.info("demo_guard: %s flagged — %s", page, detail)
            return True, f"{page.name}: {detail}"
        return False, ""
    except Exception as exc:  # noqa: BLE001 — never break a build from a guard
        logger.warning("demo_guard check failed (non-fatal): %s", exc)
        return False, ""
