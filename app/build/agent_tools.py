"""File-system tools for the build agent — sandboxed to the workspace root."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the full text content of a file inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write (create or overwrite) a file inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "content": {"type": "string", "description": "Full content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files and directories inside a workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path (use '.' for workspace root)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "create_directory",
        "description": "Create a directory (and any missing parents) inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the directory to create"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "delete_file",
        "description": "Delete a file or directory tree inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to delete"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "move_file",
        "description": "Move or rename a file or directory inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Current relative path"},
                "destination": {"type": "string", "description": "New relative path"},
            },
            "required": ["source", "destination"],
        },
    },
    {
        "name": "generate_image",
        "description": (
            "Generate an image using AI (FLUX.1) and save it to the app's assets folder. "
            "Use this for backgrounds, hero images, illustrations, icons, onboarding artwork, "
            "and any other visual asset the app needs. "
            "The saved path is returned — reference it in your code immediately after. "
            "For Flutter: also declare the assets/ directory in pubspec.yaml. "
            "For Next.js: reference as /images/<filename>. "
            "For React Native: use require('./assets/images/<filename>')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed visual description of the image. Include style, colours, "
                        "mood, and content. Example: 'A clean minimalist fitness app hero image "
                        "with a person running at sunrise, soft orange gradient background, "
                        "modern flat illustration style'"
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to save as, e.g. 'hero_background.png' or 'onboarding_1.png'",
                },
            },
            "required": ["prompt", "filename"],
        },
    },
    {
        "name": "analyze_code",
        "description": (
            "Run the real static analyzer on the workspace to find type errors, missing imports, "
            "undefined identifiers, and other issues that will cause a compilation failure. "
            "Flutter → runs 'dart analyze'. Next.js / React Native → runs 'tsc --noEmit'. "
            "Call this BEFORE reporting your validation result so real compiler errors are caught. "
            "If it returns errors, fix them with write_file before finishing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "generate_video",
        "description": (
            "Generate a short AI video (Kling Video) and save it to the app's assets folder. "
            "Use this for splash screen animations, onboarding loops, or background videos. "
            "Generation takes 30–90 seconds. The saved path is returned — reference it in your code. "
            "For Flutter: use the video_player package and declare assets/ in pubspec.yaml. "
            "For Next.js: use a <video> tag with src='/videos/<filename>'. "
            "For React Native: use expo-av or react-native-video."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed description of the video. Include motion, style, and mood. "
                        "Example: 'A smooth looping abstract blue gradient animation with "
                        "gentle flowing particles, suitable for a meditation app background'"
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to save as, e.g. 'splash.mp4' or 'onboarding_bg.mp4'",
                },
            },
            "required": ["prompt", "filename"],
        },
    },
]


def _safe(workspace: Path, rel: str) -> Path:
    """Resolve a workspace-relative path; raise ValueError on path-escape attempt."""
    resolved = (workspace / rel).resolve()
    if not str(resolved).startswith(str(workspace.resolve())):
        raise ValueError(f"Path escape attempt: {rel!r}")
    return resolved


def _detect_asset_dir(workspace: Path, media_type: str) -> Path:
    """Return (and create) the correct assets directory for the template type."""
    subdir = "images" if media_type == "image" else "videos"
    if (workspace / "pubspec.yaml").exists():
        # Flutter
        d = workspace / "assets" / subdir
    elif (workspace / "app.json").exists() or (workspace / "app.config.js").exists():
        # React Native / Expo
        d = workspace / "assets" / subdir
    else:
        # Next.js / generic web
        d = workspace / "public" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_extension(filename: str, default_ext: str) -> str:
    return filename if "." in filename else f"{filename}{default_ext}"


def _placeholder_png() -> bytes:
    """Return a minimal valid 1×1 transparent PNG (no external deps)."""
    import struct, zlib
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xcc\xcc\xcc"))  # grey pixel
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _save_placeholder(asset_dir: Path, filename: str, media_type: str) -> str:
    """Save a placeholder asset so code references compile even when generation fails."""
    out_path = asset_dir / filename
    if media_type == "image":
        out_path.write_bytes(_placeholder_png())
    else:
        out_path.write_bytes(b"")  # empty mp4 placeholder — won't play but won't break build
    return str(out_path.relative_to(asset_dir.parent.parent) if asset_dir.parent.parent.exists() else out_path)


def _run_analysis(workspace: Path, log_fn=None) -> str:
    """Run dart analyze or tsc --noEmit and return the output."""
    is_flutter = (workspace / "pubspec.yaml").exists()
    is_ts = (workspace / "tsconfig.json").exists()

    if is_flutter:
        dart = shutil.which("dart") or shutil.which("dart.exe")
        if not dart:
            return "dart not found on PATH — static analysis skipped."
        try:
            result = subprocess.run(
                [dart, "analyze", "--no-fatal-infos"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                return "dart analyze: no issues found."
            # Trim to the most useful tail
            return ("dart analyze errors:\n" + output)[-4000:]
        except subprocess.TimeoutExpired:
            return "dart analyze timed out after 120 s."
        except Exception as exc:
            return f"dart analyze failed: {exc}"

    if is_ts:
        npx = shutil.which("npx") or shutil.which("npx.cmd")
        if not npx:
            return "npx not found on PATH — TypeScript analysis skipped."
        try:
            result = subprocess.run(
                [npx, "tsc", "--noEmit", "--pretty", "false"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                return "tsc --noEmit: no type errors found."
            return ("TypeScript errors:\n" + output)[-4000:]
        except subprocess.TimeoutExpired:
            return "tsc --noEmit timed out after 120 s."
        except Exception as exc:
            return f"tsc --noEmit failed: {exc}"

    return "No recognisable project type in workspace (no pubspec.yaml or tsconfig.json)."


def _generate_image(
    prompt: str,
    filename: str,
    workspace: Path,
    log_fn=None,
) -> str:
    asset_dir = _detect_asset_dir(workspace, "image")
    filename = _ensure_extension(filename, ".png")
    out_path = asset_dir / filename
    is_flutter = (workspace / "pubspec.yaml").exists()
    flutter_note = " — declare 'assets/images/' under flutter > assets in pubspec.yaml" if is_flutter else ""

    try:
        import fal_client
        import httpx
        from app.config import get_settings

        settings = get_settings()
        if not settings.FAL_API_KEY:
            msg = "Image generation skipped — FAL_API_KEY not configured. A placeholder was saved instead."
            if log_fn:
                log_fn("warning", msg)
            out_path.write_bytes(_placeholder_png())
            rel = out_path.relative_to(workspace)
            return f"PLACEHOLDER: {rel} (no FAL key){flutter_note}. Reference this path in code — replace with real asset later."

        os.environ["FAL_KEY"] = settings.FAL_API_KEY

        result = fal_client.run(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": prompt,
                "num_images": 1,
                "image_size": "landscape_4_3",
                "num_inference_steps": 4,
            },
        )
        image_url: str = result["images"][0]["url"]

        resp = httpx.get(image_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

        rel = out_path.relative_to(workspace)
        return f"OK: image saved to {rel}{flutter_note}"

    except Exception as exc:
        raw = str(exc)
        # Detect billing / balance issues and surface them clearly
        if "exhausted balance" in raw.lower() or "locked" in raw.lower() or "billing" in raw.lower():
            user_msg = "Image generation unavailable — fal.ai account balance is exhausted. Top up at fal.ai/dashboard/billing."
        elif "unauthorized" in raw.lower() or "invalid" in raw.lower():
            user_msg = "Image generation unavailable — FAL_API_KEY is invalid. Check your .env file."
        else:
            user_msg = f"Image generation failed — saving placeholder instead. ({raw[:120]})"

        if log_fn:
            log_fn("warning", user_msg)

        # Always write a placeholder so asset references in code don't break the build
        try:
            out_path.write_bytes(_placeholder_png())
            rel = out_path.relative_to(workspace)
            return f"PLACEHOLDER: {rel} ({user_msg}){flutter_note}. Use this path in code — the image will be grey until generation works."
        except Exception:
            return f"ERROR: {user_msg}"


def _generate_video(
    prompt: str,
    filename: str,
    workspace: Path,
    log_fn=None,
) -> str:
    asset_dir = _detect_asset_dir(workspace, "video")
    filename = _ensure_extension(filename, ".mp4")
    out_path = asset_dir / filename
    is_flutter = (workspace / "pubspec.yaml").exists()
    flutter_note = " — declare 'assets/videos/' under flutter > assets in pubspec.yaml" if is_flutter else ""

    try:
        import fal_client
        import httpx
        from app.config import get_settings

        settings = get_settings()
        if not settings.FAL_API_KEY:
            msg = "Video generation skipped — FAL_API_KEY not configured. A placeholder was saved."
            if log_fn:
                log_fn("warning", msg)
            out_path.write_bytes(b"")
            rel = out_path.relative_to(workspace)
            return f"PLACEHOLDER: {rel} (no FAL key){flutter_note}. Reference this path in code."

        os.environ["FAL_KEY"] = settings.FAL_API_KEY

        result = fal_client.run(
            "fal-ai/kling-video/v1.6/standard/text-to-video",
            arguments={
                "prompt": prompt,
                "duration": "5",
                "aspect_ratio": "16:9",
            },
        )
        video_url: str = result["video"]["url"]

        resp = httpx.get(video_url, timeout=300, follow_redirects=True)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

        rel = out_path.relative_to(workspace)
        return f"OK: video saved to {rel}{flutter_note}"

    except Exception as exc:
        raw = str(exc)
        if "exhausted balance" in raw.lower() or "locked" in raw.lower() or "billing" in raw.lower():
            user_msg = "Video generation unavailable — fal.ai account balance is exhausted. Top up at fal.ai/dashboard/billing."
        elif "unauthorized" in raw.lower() or "invalid" in raw.lower():
            user_msg = "Video generation unavailable — FAL_API_KEY is invalid."
        else:
            user_msg = f"Video generation failed — saving placeholder instead. ({raw[:120]})"

        if log_fn:
            log_fn("warning", user_msg)

        try:
            out_path.write_bytes(b"")
            rel = out_path.relative_to(workspace)
            return f"PLACEHOLDER: {rel} ({user_msg}){flutter_note}. Use this path in code."
        except Exception:
            return f"ERROR: {user_msg}"


def execute_tool(
    name: str,
    inputs: dict[str, Any],
    workspace: Path,
    log_fn=None,
) -> str:
    """Dispatch a tool call and return the result as a string."""
    match name:
        case "read_file":
            p = _safe(workspace, inputs["path"])
            if not p.exists():
                return f"ERROR: {inputs['path']} not found"
            return p.read_text(encoding="utf-8", errors="replace")

        case "write_file":
            p = _safe(workspace, inputs["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inputs["content"], encoding="utf-8")
            return f"OK: wrote {len(inputs['content'])} chars to {inputs['path']}"

        case "list_files":
            p = _safe(workspace, inputs["path"])
            if not p.exists():
                return f"ERROR: {inputs['path']} not found"
            entries = sorted(os.listdir(p))
            if not entries:
                return "(empty)"
            lines = []
            for entry in entries:
                suffix = "/" if (p / entry).is_dir() else ""
                lines.append(f"{entry}{suffix}")
            return "\n".join(lines)

        case "create_directory":
            p = _safe(workspace, inputs["path"])
            p.mkdir(parents=True, exist_ok=True)
            return f"OK: created {inputs['path']}"

        case "delete_file":
            p = _safe(workspace, inputs["path"])
            if not p.exists():
                return f"ERROR: {inputs['path']} not found"
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            return f"OK: deleted {inputs['path']}"

        case "move_file":
            src = _safe(workspace, inputs["source"])
            dst = _safe(workspace, inputs["destination"])
            if not src.exists():
                return f"ERROR: {inputs['source']} not found"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return f"OK: moved {inputs['source']} → {inputs['destination']}"

        case "analyze_code":
            return _run_analysis(workspace, log_fn)

        case "generate_image":
            return _generate_image(inputs["prompt"], inputs["filename"], workspace, log_fn)

        case "generate_video":
            return _generate_video(inputs["prompt"], inputs["filename"], workspace, log_fn)

        case _:
            return f"ERROR: unknown tool {name!r}"
