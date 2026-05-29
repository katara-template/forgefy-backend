"""File-system tools for the build agent — sandboxed to the workspace root."""
from __future__ import annotations

import os
import shutil
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


def _generate_image(prompt: str, filename: str, workspace: Path) -> str:
    try:
        import fal_client
        import httpx
        from app.config import get_settings

        settings = get_settings()
        if not settings.FAL_API_KEY:
            return "ERROR: FAL_API_KEY not configured — skipping image generation"

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

        asset_dir = _detect_asset_dir(workspace, "image")
        filename = _ensure_extension(filename, ".png")
        out_path = asset_dir / filename

        resp = httpx.get(image_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

        rel = out_path.relative_to(workspace)
        is_flutter = (workspace / "pubspec.yaml").exists()
        flutter_note = " — remember to declare 'assets/images/' under flutter > assets in pubspec.yaml" if is_flutter else ""
        return f"OK: image saved to {rel}{flutter_note}"

    except Exception as exc:
        return f"ERROR: image generation failed: {exc}"


def _generate_video(prompt: str, filename: str, workspace: Path) -> str:
    try:
        import fal_client
        import httpx
        from app.config import get_settings

        settings = get_settings()
        if not settings.FAL_API_KEY:
            return "ERROR: FAL_API_KEY not configured — skipping video generation"

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

        asset_dir = _detect_asset_dir(workspace, "video")
        filename = _ensure_extension(filename, ".mp4")
        out_path = asset_dir / filename

        resp = httpx.get(video_url, timeout=300, follow_redirects=True)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

        rel = out_path.relative_to(workspace)
        is_flutter = (workspace / "pubspec.yaml").exists()
        flutter_note = " — remember to declare 'assets/videos/' under flutter > assets in pubspec.yaml" if is_flutter else ""
        return f"OK: video saved to {rel}{flutter_note}"

    except Exception as exc:
        return f"ERROR: video generation failed: {exc}"


def execute_tool(name: str, inputs: dict[str, Any], workspace: Path) -> str:
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

        case "generate_image":
            return _generate_image(inputs["prompt"], inputs["filename"], workspace)

        case "generate_video":
            return _generate_video(inputs["prompt"], inputs["filename"], workspace)

        case _:
            return f"ERROR: unknown tool {name!r}"
