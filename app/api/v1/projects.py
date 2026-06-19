"""Projects endpoints — list, get, and dispatch prompt-driven updates."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter

from app.core.exceptions import ForbiddenError, NotFoundError, ValidationError
from app.deps import CurrentUser, DBSession
from app.schemas.project import ProjectOut, UpdateProjectRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _doc_to_out(doc) -> ProjectOut:
    d = doc.to_dict()
    return ProjectOut(
        id=uuid.UUID(doc.id),
        owner_id=uuid.UUID(d["owner_id"]),
        app_name=d["app_name"],
        template_key=d["template_key"],
        repo_full_name=d["repo_full_name"],
        github_url=d["github_url"],
        repo_owner=d.get("repo_owner"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        session_id=uuid.UUID(d["session_id"]) if d.get("session_id") else None,
        blueprint_id=uuid.UUID(d["blueprint_id"]) if d.get("blueprint_id") else None,
        preview_url=d.get("preview_url"),
        artifact_url=d.get("artifact_url"),
        is_updating=d.get("is_updating", False),
        build_error=d.get("build_error"),
        build_error_action=d.get("build_error_action"),
    )


async def _get_owned(project_id: uuid.UUID, user_id: uuid.UUID, db) -> ProjectOut:
    doc = await db.collection("projects").document(str(project_id)).get()
    if not doc.exists:
        raise NotFoundError(f"Project {project_id} not found")
    d = doc.to_dict()
    if uuid.UUID(d["owner_id"]) != user_id:
        raise ForbiddenError("Access denied")
    return _doc_to_out(doc)


@router.get("", response_model=list[ProjectOut])
async def list_projects(db: DBSession, user: CurrentUser) -> list[ProjectOut]:
    """Return all projects owned by the current user."""
    docs = (
        await db.collection("projects")
        .where("owner_id", "==", str(user.id))
        .get()
    )
    projects = sorted(
        [_doc_to_out(d) for d in docs],
        key=lambda p: p.updated_at,
        reverse=True,
    )
    return projects


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> ProjectOut:
    """Return a single project."""
    return await _get_owned(project_id, user.id, db)


@router.post("/{project_id}/update", response_model=dict)
async def update_project(
    project_id: uuid.UUID,
    body: UpdateProjectRequest,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Dispatch a prompt-driven update for the project. Returns immediately."""
    project = await _get_owned(project_id, user.id, db)

    if project.is_updating:
        raise ValidationError("A build or update is already in progress.")

    from app.workers.update_worker import apply_update
    apply_update.apply_async(
        args=[str(project_id), body.prompt, str(user.id)],
        queue="build",
    )
    return {"status": "queued"}


@router.post("/{project_id}/build-preview", response_model=dict)
async def trigger_preview_build(
    project_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Manually trigger a preview build and deployment for the project."""
    project = await _get_owned(project_id, user.id, db)

    if project.is_updating:
        raise ValidationError("A build or update is already in progress.")

    if not project.github_url:
        raise ValidationError("Project has no GitHub repository yet — wait for the initial build to finish.")

    from app.workers.build_worker import build_preview
    build_preview.apply_async(
        args=[str(project_id), str(user.id)],
        queue="build",
    )
    return {"status": "queued"}
