"""Admin endpoints — system-wide configuration (build model, etc.)."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.build_model import VALID_BUILD_MODELS
from app.core.exceptions import ValidationError
from app.core.tiers import TIERS
from app.deps import AdminUser, DBSession

router = APIRouter()

TEMPLATE_LABELS: dict[str, str] = {
    "flutter": "Flutter",
    "react_native": "React Native",
    "next": "Next.js",
}

# Sessions still in the meeting → blueprint phase. Once approved, progress is
# tracked via the project's is_updating/preview_url instead (see apps_building/
# apps_shipped below) so a session isn't counted as "in progress" twice.
_IN_PROGRESS_SESSION_STATUSES = ["WAITING", "JOINING", "LISTENING", "PROCESSING", "BLUEPRINT_READY"]


class BuildModelResponse(BaseModel):
    model: str


class SetBuildModelRequest(BaseModel):
    model: str


@router.get("/build-model", response_model=BuildModelResponse)
async def get_build_model_setting(db: DBSession, user: AdminUser) -> BuildModelResponse:
    """Return the active build model (Firestore override takes precedence over .env)."""
    from app.config import get_settings

    settings = get_settings()
    doc = await db.collection("system").document("config").get()
    model = (doc.to_dict() or {}).get("build_model") if doc.exists else None
    return BuildModelResponse(model=model or settings.BUILD_MODEL)


@router.patch("/build-model", response_model=BuildModelResponse)
async def set_build_model_setting(
    body: SetBuildModelRequest,
    db: DBSession,
    user: AdminUser,
) -> BuildModelResponse:
    """Persist a new build model to Firestore (takes effect on the next build/update)."""
    if body.model not in VALID_BUILD_MODELS:
        raise ValidationError(f"Invalid model '{body.model}'. Choose from: {', '.join(VALID_BUILD_MODELS)}")

    await db.collection("system").document("config").set(
        {"build_model": body.model}, merge=True
    )
    return BuildModelResponse(model=body.model)


class AlertOut(BaseModel):
    id: str
    title: str
    raw_detail: str
    source: str
    session_id: str | None = None
    project_id: str | None = None
    resolved: bool
    created_at: datetime


class ResolveAlertRequest(BaseModel):
    resolved: bool = True


@router.get("/alerts", response_model=list[AlertOut])
async def list_alerts(
    db: DBSession,
    user: AdminUser,
    resolved: bool = False,
    limit: int = 50,
) -> list[AlertOut]:
    """List operator alerts (support-tier build failures), newest first.

    Sorts in Python rather than via Firestore's order_by — a where() filter
    combined with order_by on a different field needs a composite index, which
    isn't guaranteed to exist in every environment (see firestore.indexes.json).
    Operator alerts are low-volume, so fetching the filtered set and sorting
    locally is cheap and avoids that dependency entirely.
    """
    docs = await db.collection("operator_alerts").where("resolved", "==", resolved).get()
    alerts = [AlertOut(id=d.id, **d.to_dict()) for d in docs]
    alerts.sort(key=lambda a: a.created_at, reverse=True)
    return alerts[:limit]


@router.patch("/alerts/{alert_id}", response_model=dict)
async def resolve_alert(
    alert_id: str,
    body: ResolveAlertRequest,
    db: DBSession,
    user: AdminUser,
) -> dict:
    """Mark an operator alert resolved (or unresolved)."""
    await db.collection("operator_alerts").document(alert_id).update({"resolved": body.resolved})
    return {"ok": True}


class PipelineStats(BaseModel):
    meetings_in_progress: int
    apps_building: int
    apps_shipped: int


class OverviewStats(BaseModel):
    meetings_all_time: int
    meetings_this_month: int
    apps_total: int
    apps_by_framework: dict[str, int]
    active_users: int
    new_signups_week: int
    pipeline: PipelineStats


class ActivityItem(BaseModel):
    id: str
    title: str
    action: str
    framework: str
    status: str
    updated_at: datetime


class OverviewOut(BaseModel):
    stats: OverviewStats
    recent_activity: list[ActivityItem]


async def _count(query) -> int:
    """Return a Firestore aggregation count for a collection or query.

    Far cheaper than fetching every matching document just to call len() —
    the aggregation only tallies matches server-side.
    """
    result = await query.count().get()
    return int(result[0][0].value)


def _activity_from_project(doc) -> ActivityItem:
    """Derive a human activity label/status from a project doc's current state.

    Best-effort v1: sourced from projects only, so meeting-processing-only
    activity (session states before a project exists) isn't represented yet.
    """
    d = doc.to_dict()
    framework = TEMPLATE_LABELS.get(d.get("template_key", ""), d.get("template_key", ""))
    if d.get("build_error"):
        action, status = "Build failed", "failed"
    elif d.get("is_updating"):
        action, status = "Build started", "building"
    elif d.get("preview_url"):
        action, status = "App shipped", "success"
    else:
        action, status = "Build queued", "queued"
    return ActivityItem(
        id=doc.id,
        title=d.get("app_name", ""),
        action=action,
        framework=framework,
        status=status,
        updated_at=d["updated_at"],
    )


@router.get("/overview", response_model=OverviewOut)
async def get_overview(db: DBSession, user: AdminUser) -> OverviewOut:
    """Platform-wide stats + recent activity for the dashboard's Overview page.

    All counts are computed live via Firestore aggregation queries (see _count)
    rather than a precomputed snapshot — simple and always accurate, and cheap
    even at scale since aggregation queries don't read full documents.
    """
    now = datetime.now(UTC)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    week_ago = now - timedelta(days=7)

    sessions = db.collection("sessions")
    projects = db.collection("projects")
    users = db.collection("users")

    # Every count is independent, so run them all concurrently — one
    # round-trip's worth of latency instead of a dozen sequential ones.
    (
        framework_counts,
        recent_docs,
        meetings_all_time,
        meetings_this_month,
        apps_total,
        active_users,
        new_signups_week,
        meetings_in_progress,
        apps_building,
        apps_shipped,
    ) = await asyncio.gather(
        asyncio.gather(
            *(_count(projects.where("template_key", "==", key)) for key in TEMPLATE_LABELS)
        ),
        projects.order_by("updated_at", direction="DESCENDING").limit(10).get(),
        _count(sessions),
        _count(sessions.where("created_at", ">=", month_start)),
        _count(projects),
        _count(users),
        _count(users.where("created_at", ">=", week_ago)),
        _count(sessions.where("status", "in", _IN_PROGRESS_SESSION_STATUSES)),
        _count(projects.where("is_updating", "==", True)),
        _count(projects.where("preview_url", "!=", None)),
    )

    stats = OverviewStats(
        meetings_all_time=meetings_all_time,
        meetings_this_month=meetings_this_month,
        apps_total=apps_total,
        apps_by_framework=dict(zip(TEMPLATE_LABELS.values(), framework_counts)),
        active_users=active_users,
        new_signups_week=new_signups_week,
        pipeline=PipelineStats(
            meetings_in_progress=meetings_in_progress,
            apps_building=apps_building,
            apps_shipped=apps_shipped,
        ),
    )

    return OverviewOut(
        stats=stats,
        recent_activity=[_activity_from_project(d) for d in recent_docs],
    )


_PLATFORMS = ("meet", "zoom", "teams", "physical")

# A session's own status describes whether ITS meeting-to-blueprint work
# succeeded — independent of whether the resulting app build later failed.
_FAILED_SESSION_STATUSES = ["FAILED"]
_PROCESSED_SESSION_STATUSES = ["BLUEPRINT_READY", "APPROVED", "BUILDING"]
_PENDING_SESSION_STATUSES = ["WAITING", "JOINING", "LISTENING", "PROCESSING"]


class MeetingsByStatus(BaseModel):
    processed: int
    pending: int
    failed: int


class DailyCount(BaseModel):
    date: str  # YYYY-MM-DD
    count: int


class MeetingsStatsOut(BaseModel):
    total: int
    this_month: int
    by_status: MeetingsByStatus
    by_platform: dict[str, int]
    per_day: list[DailyCount]


@router.get("/meetings/stats", response_model=MeetingsStatsOut)
async def get_meetings_stats(db: DBSession, user: AdminUser) -> MeetingsStatsOut:
    """Aggregate-only meeting statistics — deliberately exposes no individual
    meeting record (no titles, no per-meeting dates, no session identifiers).
    Every number here is a Firestore count() aggregation or a same-day tally,
    never a read of a specific user's meeting content.
    """
    now = datetime.now(UTC)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    sessions = db.collection("sessions")

    day_starts = [
        (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(29, -1, -1)
    ]

    # All counts (30 daily buckets, per-platform, per-status, totals) are
    # independent aggregations — run them concurrently rather than as ~40
    # sequential round trips.
    (
        platform_counts,
        day_counts,
        total,
        this_month,
        processed,
        pending,
        failed,
    ) = await asyncio.gather(
        asyncio.gather(
            *(_count(sessions.where("platform", "==", platform)) for platform in _PLATFORMS)
        ),
        asyncio.gather(
            *(
                _count(
                    sessions
                    .where("created_at", ">=", day_start)
                    .where("created_at", "<", day_start + timedelta(days=1))
                )
                for day_start in day_starts
            )
        ),
        _count(sessions),
        _count(sessions.where("created_at", ">=", month_start)),
        _count(sessions.where("status", "in", _PROCESSED_SESSION_STATUSES)),
        _count(sessions.where("status", "in", _PENDING_SESSION_STATUSES)),
        _count(sessions.where("status", "in", _FAILED_SESSION_STATUSES)),
    )

    return MeetingsStatsOut(
        total=total,
        this_month=this_month,
        by_status=MeetingsByStatus(processed=processed, pending=pending, failed=failed),
        by_platform=dict(zip(_PLATFORMS, platform_counts)),
        per_day=[
            DailyCount(date=day_start.date().isoformat(), count=count)
            for day_start, count in zip(day_starts, day_counts)
        ],
    )


class BuildOut(BaseModel):
    name: str
    description: str
    artifact_url: str | None = None


@router.get("/builds", response_model=list[BuildOut])
async def list_builds(db: DBSession, user: AdminUser, limit: int = 200) -> list[BuildOut]:
    """List generated apps — name, description, and artifact link only.

    No meeting/session linkage, framework, status, or timestamp is exposed —
    just the app itself. `description` is pulled from the project's linked
    blueprint (a direct point lookup by blueprint_id, not a query).
    """
    docs = (
        await db.collection("projects")
        .order_by("created_at", direction="DESCENDING")
        .limit(limit)
        .get()
    )
    dicts = [doc.to_dict() for doc in docs]

    # Batch-fetch all linked blueprints in one get_all call instead of one
    # sequential point read per project.
    blueprint_refs = {
        bid: db.collection("blueprints").document(bid)
        for d in dicts
        if (bid := d.get("blueprint_id"))
    }
    descriptions: dict[str, str] = {}
    if blueprint_refs:
        async for bp_doc in db.get_all(list(blueprint_refs.values())):
            if bp_doc.exists:
                descriptions[bp_doc.id] = (
                    (bp_doc.to_dict().get("json_output") or {}).get("app_description", "")
                )

    return [
        BuildOut(
            name=d.get("app_name", ""),
            description=descriptions.get(d.get("blueprint_id", ""), ""),
            artifact_url=d.get("artifact_url"),
        )
        for d in dicts
    ]


class UserOut(BaseModel):
    email: str
    tier: str
    joined: datetime
    meetings_count: int
    apps_count: int
    tokens_used_this_month: int
    monthly_token_budget: int


@router.get("/users", response_model=list[UserOut])
async def list_users(db: DBSession, user: AdminUser, limit: int = 50) -> list[UserOut]:
    """List registered users — account/billing info and aggregate counts only.

    meetings_count/apps_count are Firestore count() aggregations (per-user
    tallies), never a fetch of the meetings/apps themselves — no meeting
    content of any kind reaches this response. tokens_used_this_month lets an
    admin verify usage is actually accumulating for a given account (e.g. to
    check why a user hasn't hit their tier's quota yet) without exposing what
    they actually built it for.
    """
    from app.core.tiers import get_tier
    from app.core.usage import get_monthly_tokens

    docs = (
        await db.collection("users")
        .order_by("created_at", direction="DESCENDING")
        .limit(limit)
        .get()
    )

    async def _user_out(doc) -> UserOut:
        d = doc.to_dict()
        uid = doc.id
        tier_key = d.get("tier", "free")
        meetings_count, apps_count, tokens_used = await asyncio.gather(
            _count(db.collection("sessions").where("user_id", "==", uid)),
            _count(db.collection("projects").where("owner_id", "==", uid)),
            get_monthly_tokens(db, uid),
        )
        return UserOut(
            email=d.get("email", ""),
            tier=tier_key,
            joined=d["created_at"],
            meetings_count=meetings_count,
            apps_count=apps_count,
            tokens_used_this_month=tokens_used,
            monthly_token_budget=get_tier(tier_key).monthly_tokens,
        )

    return list(await asyncio.gather(*(_user_out(doc) for doc in docs)))


class TrendPoint(BaseModel):
    date: str  # YYYY-MM-DD
    meetings: int
    apps_built: int


class FrameworkPerformance(BaseModel):
    framework: str
    total: int
    shipped: int
    failed: int
    success_rate: float  # 0-100, 0 if total is 0


class TierBreakdown(BaseModel):
    tier: str
    count: int
    price_usd: int


class SubscriptionStats(BaseModel):
    by_tier: list[TierBreakdown]
    mrr_usd: int
    free_users: int
    paid_users: int
    conversion_rate: float  # paid / total * 100, 0 if no users


class ProjectPerformance(BaseModel):
    total: int
    shipped: int
    building: int
    failed: int
    success_rate: float  # 0-100, 0 if total is 0


class FunnelStage(BaseModel):
    stage: str
    count: int


class AnalyticsOut(BaseModel):
    trend: list[TrendPoint]
    frameworks: list[FrameworkPerformance]
    subscriptions: SubscriptionStats
    project_performance: ProjectPerformance
    funnel: list[FunnelStage]


async def _framework_performance(db, framework_key: str, label: str) -> FrameworkPerformance:
    """Success rate per framework — fetched and tallied in Python rather than a
    second server-side filter, since combining an equality filter (template_key)
    with an inequality filter (preview_url/build_error) on a different field
    would need a composite index that isn't guaranteed to exist (see
    firestore.indexes.json and list_alerts's docstring for the same tradeoff).
    Nothing here is returned to the client except the tallies.
    """
    docs = await db.collection("projects").where("template_key", "==", framework_key).get()
    total = len(docs)
    shipped = sum(1 for d in docs if d.to_dict().get("preview_url"))
    failed = sum(1 for d in docs if d.to_dict().get("build_error"))
    return FrameworkPerformance(
        framework=label,
        total=total,
        shipped=shipped,
        failed=failed,
        success_rate=round(shipped / total * 100, 1) if total else 0.0,
    )


@router.get("/analytics", response_model=AnalyticsOut)
async def get_analytics(db: DBSession, user: AdminUser) -> AnalyticsOut:
    """Platform-wide analytics — trend, framework performance, subscriptions,
    project performance, and a conversion funnel. Every number here is a
    Firestore count() aggregation (or, for per-framework success rate, a tally
    over a single-field-filtered fetch) — no individual meeting, project, or
    user record is ever included in the response.
    """
    now = datetime.now(UTC)
    sessions = db.collection("sessions")
    projects = db.collection("projects")
    users = db.collection("users")
    blueprints = db.collection("blueprints")

    async def _day_point(i: int) -> TrendPoint:
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        meetings, apps_built = await asyncio.gather(
            _count(sessions.where("created_at", ">=", day_start).where("created_at", "<", day_end)),
            _count(projects.where("created_at", ">=", day_start).where("created_at", "<", day_end)),
        )
        return TrendPoint(date=day_start.date().isoformat(), meetings=meetings, apps_built=apps_built)

    trend_task = asyncio.gather(*(_day_point(i) for i in range(29, -1, -1)))
    frameworks_task = asyncio.gather(
        *(_framework_performance(db, key, label) for key, label in TEMPLATE_LABELS.items())
    )
    tier_counts_task = asyncio.gather(
        *(_count(users.where("tier", "==", tier_key)) for tier_key in TIERS)
    )
    project_counts_task = asyncio.gather(
        _count(projects),
        _count(projects.where("is_updating", "==", True)),
        _count(projects.where("preview_url", "!=", None)),
        _count(projects.where("build_error", "!=", None)),
    )
    funnel_counts_task = asyncio.gather(
        _count(sessions),
        _count(blueprints),
        _count(blueprints.where("approved", "==", True)),
        _count(projects.where("preview_url", "!=", None)),
    )

    trend, framework_perf, tier_counts, project_counts, funnel_counts = await asyncio.gather(
        trend_task, frameworks_task, tier_counts_task, project_counts_task, funnel_counts_task
    )

    tier_breakdown = [
        TierBreakdown(tier=tier_key, count=count, price_usd=TIERS[tier_key].price_usd)
        for tier_key, count in zip(TIERS.keys(), tier_counts, strict=True)
    ]
    free_users = next((t.count for t in tier_breakdown if t.tier == "free"), 0)
    total_users = sum(t.count for t in tier_breakdown)
    paid_users = total_users - free_users
    mrr_usd = sum(t.count * t.price_usd for t in tier_breakdown)

    total_projects, building, shipped, failed = project_counts

    return AnalyticsOut(
        trend=trend,
        frameworks=framework_perf,
        subscriptions=SubscriptionStats(
            by_tier=tier_breakdown,
            mrr_usd=mrr_usd,
            free_users=free_users,
            paid_users=paid_users,
            conversion_rate=round(paid_users / total_users * 100, 1) if total_users else 0.0,
        ),
        project_performance=ProjectPerformance(
            total=total_projects,
            shipped=shipped,
            building=building,
            failed=failed,
            success_rate=round(shipped / total_projects * 100, 1) if total_projects else 0.0,
        ),
        funnel=[
            FunnelStage(stage="Meetings", count=funnel_counts[0]),
            FunnelStage(stage="Blueprints created", count=funnel_counts[1]),
            FunnelStage(stage="Blueprints approved", count=funnel_counts[2]),
            FunnelStage(stage="Apps shipped", count=funnel_counts[3]),
        ],
    )
