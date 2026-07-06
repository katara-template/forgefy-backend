"""Blueprint endpoints: get and approve.

Firestore collections used:
  blueprints/{blueprint_id}  — session_id, json_output, approved, created_at
  sessions/{session_id}      — user_id, status, ...
"""
import logging
import re
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter
from google.cloud.firestore import AsyncClient
from pydantic import BaseModel

from app.core.exceptions import ForbiddenError, NotFoundError, ValidationError
from app.db.models.blueprint import Blueprint
from app.db.models.enums import SessionStatus
from app.deps import CurrentUser, DBSession
from app.modules.voxa.state_machine import MeetingStateMachine
from app.schemas.blueprint import BlueprintOut

logger = logging.getLogger(__name__)
router = APIRouter()


def _doc_to_blueprint(doc) -> Blueprint:
    data = doc.to_dict()
    return Blueprint(
        id=uuid.UUID(doc.id),
        session_id=uuid.UUID(data["session_id"]),
        json_output=data.get("json_output"),
        approved=data.get("approved", False),
        created_at=data["created_at"],
        repo_url=data.get("repo_url"),
        repo_name=data.get("repo_name"),
        build_summary=data.get("build_summary"),
        build_status=data.get("build_status"),
        artifact_url=data.get("artifact_url"),
        preview_url=data.get("preview_url"),
        project_id=data.get("project_id"),
    )


async def _get_owned_blueprint(
    blueprint_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncClient,
) -> Blueprint:
    """Load blueprint; raise 404/403 if missing or session is unowned."""
    doc = await db.collection("blueprints").document(str(blueprint_id)).get()
    if not doc.exists:
        raise NotFoundError(f"Blueprint {blueprint_id} not found")

    blueprint = _doc_to_blueprint(doc)

    sess_doc = await db.collection("sessions").document(str(blueprint.session_id)).get()
    if not sess_doc.exists or uuid.UUID(sess_doc.to_dict()["user_id"]) != user_id:
        raise ForbiddenError("Access denied")

    return blueprint


@router.get("/session/{session_id}/all", response_model=list[BlueprintOut])
async def list_blueprints_by_session(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> list[BlueprintOut]:
    """Return all blueprints for a session sorted newest-first (ownership-checked)."""
    sess_doc = await db.collection("sessions").document(str(session_id)).get()
    if not sess_doc.exists or uuid.UUID(sess_doc.to_dict()["user_id"]) != user.id:
        raise NotFoundError(f"Session {session_id} not found")

    docs = (
        await db.collection("blueprints")
        .where("session_id", "==", str(session_id))
        .get()
    )
    sorted_docs = sorted(docs, key=lambda d: d.to_dict().get("created_at", ""), reverse=True)
    return [BlueprintOut.model_validate(_doc_to_blueprint(d)) for d in sorted_docs]


@router.get("/session/{session_id}", response_model=BlueprintOut)
async def get_blueprint_by_session(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> BlueprintOut:
    """Return the latest blueprint for a session (ownership-checked)."""
    sess_doc = await db.collection("sessions").document(str(session_id)).get()
    if not sess_doc.exists or uuid.UUID(sess_doc.to_dict()["user_id"]) != user.id:
        raise NotFoundError(f"Session {session_id} not found")

    docs = (
        await db.collection("blueprints")
        .where("session_id", "==", str(session_id))
        .get()
    )
    if not docs:
        raise NotFoundError(f"No blueprint found for session {session_id}")

    latest = max(docs, key=lambda d: d.to_dict().get("created_at", ""))
    return BlueprintOut.model_validate(_doc_to_blueprint(latest))


@router.get("/{blueprint_id}", response_model=BlueprintOut)
async def get_blueprint(
    blueprint_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> BlueprintOut:
    """Retrieve a blueprint by ID."""
    blueprint = await _get_owned_blueprint(blueprint_id, user.id, db)
    return BlueprintOut.model_validate(blueprint)


class BlueprintPatchRequest(BaseModel):
    app_name: str | None = None
    template_key: str | None = None
    features: list[dict] | None = None


@router.patch("/{blueprint_id}", response_model=BlueprintOut)
async def patch_blueprint(
    blueprint_id: uuid.UUID,
    body: BlueprintPatchRequest,
    db: DBSession,
    user: CurrentUser,
) -> BlueprintOut:
    """Edit app_name, template, or features before approving a blueprint."""
    blueprint = await _get_owned_blueprint(blueprint_id, user.id, db)

    if blueprint.approved:
        raise ValidationError("Cannot edit an already-approved blueprint")

    current: dict = dict(blueprint.json_output or {})

    if body.app_name is not None:
        current["app_name"] = body.app_name
    if body.template_key is not None:
        current["template"] = body.template_key
    if body.features is not None:
        current["features"] = body.features

    await db.collection("blueprints").document(str(blueprint_id)).update({"json_output": current})
    blueprint.json_output = current
    return BlueprintOut.model_validate(blueprint)


@router.post("/{blueprint_id}/approve", response_model=BlueprintOut)
async def approve_blueprint(
    blueprint_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> BlueprintOut:
    """Approve a blueprint and dispatch the build task.

    Idempotent: if the session is already APPROVED (e.g. a previous build task
    crashed before starting), the state transition is skipped and the build is
    re-dispatched so the user can retry without manual intervention.
    """
    from app.core.usage import check_not_over_limit
    await check_not_over_limit(db, str(user.id))

    blueprint = await _get_owned_blueprint(blueprint_id, user.id, db)

    sess_doc = await db.collection("sessions").document(str(blueprint.session_id)).get()
    current_status = sess_doc.to_dict().get("status", "") if sess_doc.exists else ""

    if current_status != SessionStatus.APPROVED.value:
        sm = MeetingStateMachine(db)
        await sm.transition(
            blueprint.session_id,
            SessionStatus.APPROVED,
            extra_payload={"blueprint_id": str(blueprint_id), "approved_by": str(user.id)},
        )

    # Reuse an existing project_id if the build is being retried, otherwise mint one now
    # so the frontend can navigate to the project page before the task starts.
    project_id = blueprint.project_id or str(uuid.uuid4())

    bp_updates: dict = {"approved": True, "project_id": project_id}
    if not blueprint.approved:
        await db.collection("blueprints").document(str(blueprint_id)).update(bp_updates)
        blueprint.approved = True
    elif not blueprint.project_id:
        await db.collection("blueprints").document(str(blueprint_id)).update({"project_id": project_id})

    blueprint.project_id = project_id

    # Create the project stub now so the project page loads immediately.
    # The build task overwrites this with full data as the build progresses.
    json_out = blueprint.json_output or {}
    raw_name = json_out.get("app_name") or (json_out.get("app_description") or "")[:40] or f"app-{str(blueprint.session_id)[:8]}"
    stub_app_name = re.sub(r"[^a-zA-Z0-9._-]", "-", raw_name).strip("-").lower()[:100] or "forgefy-app"
    stub_template = json_out.get("template", "next")
    now = datetime.now(UTC)

    # Never provision/wire a database on our own initiative — ask first. See
    # app/build/blueprint_generator.py's classify_needs_database and the project
    # page's handling of db_decision_pending (app/api/v1/projects.py).
    from app.build.blueprint_generator import classify_needs_database
    needs_db, db_reason = await classify_needs_database(
        json_out.get("app_description", ""), json_out.get("features", [])
    )

    await db.collection("projects").document(project_id).set({
        "owner_id": str(user.id),
        "session_id": str(blueprint.session_id),
        "blueprint_id": str(blueprint_id),
        "app_name": stub_app_name,
        "template_key": stub_template,
        "repo_full_name": "",
        "github_url": "",
        "preview_url": None,
        "artifact_url": None,
        "is_updating": not needs_db,
        "build_error": None,
        "blueprint_context": json_out,
        "db_decision_pending": needs_db,
        "db_decision_reason": db_reason if needs_db else None,
        "created_at": now,
        "updated_at": now,
    })

    if needs_db:
        logger.info(
            "Build withheld pending database decision session=%s blueprint=%s project=%s",
            blueprint.session_id, blueprint_id, project_id,
        )
    else:
        from app.workers.build_worker import run_build
        run_build.apply_async(args=[str(blueprint.session_id), project_id], queue="build")
        logger.info("Build dispatched session=%s blueprint=%s project=%s", blueprint.session_id, blueprint_id, project_id)

    return BlueprintOut.model_validate(blueprint)
