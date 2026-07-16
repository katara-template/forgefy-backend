"""API key management endpoints (dashboard-facing, JWT-authed).

Developers create keys here, then authenticate to the developer API
(/v1/extract, …) with `Authorization: Bearer fgy_live_…` — see
app.deps.get_api_key. The raw key is returned exactly once at creation;
only its SHA-256 hash is stored.

Firestore collections used:
  api_keys/{key_id} — owner_user_id, name, prefix, key_hash, created_at,
                      last_used_at, revoked_at
"""
import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Request

from app.core.api_keys import display_prefix, generate_api_key, hash_api_key
from app.core.exceptions import NotFoundError, ValidationError
from app.core.rate_limit import limiter
from app.deps import CurrentUser, DBSession
from app.schemas.api_key import ApiKeyCreatedResponse, ApiKeyOut, CreateApiKeyRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# Enough for any sane rotation scheme; a cap mostly guards against a runaway
# script minting keys in a loop.
_MAX_ACTIVE_KEYS_PER_USER = 10


@router.post("", response_model=ApiKeyCreatedResponse, status_code=201)
@limiter.limit("60/minute")
async def create_api_key(
    request: Request,
    body: CreateApiKeyRequest,
    db: DBSession,
    user: CurrentUser,
) -> ApiKeyCreatedResponse:
    """Create an API key. The full key appears in this response and never again."""
    docs = await db.collection("api_keys").where("owner_user_id", "==", str(user.id)).get()
    active = sum(1 for d in docs if not (d.to_dict() or {}).get("revoked_at"))
    if active >= _MAX_ACTIVE_KEYS_PER_USER:
        raise ValidationError(
            f"Active API key limit reached ({_MAX_ACTIVE_KEYS_PER_USER}). "
            "Revoke an unused key first."
        )

    key = generate_api_key()
    key_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    await db.collection("api_keys").document(key_id).set({
        "owner_user_id": str(user.id),
        "name": body.name,
        "prefix": display_prefix(key),
        "key_hash": hash_api_key(key),
        "created_at": now,
        "last_used_at": None,
        "revoked_at": None,
    })

    logger.info("API key created: id=%s user=%s", key_id, user.id)
    return ApiKeyCreatedResponse(
        id=uuid.UUID(key_id),
        name=body.name,
        prefix=display_prefix(key),
        created_at=now,
        key=key,
    )


@router.get("", response_model=list[ApiKeyOut])
@limiter.limit("60/minute")
async def list_api_keys(
    request: Request,
    db: DBSession,
    user: CurrentUser,
) -> list[ApiKeyOut]:
    """List the current user's API keys (revoked ones included), newest first."""
    docs = await db.collection("api_keys").where("owner_user_id", "==", str(user.id)).get()
    keys = [
        ApiKeyOut(
            id=uuid.UUID(doc.id),
            name=data.get("name", ""),
            prefix=data.get("prefix", ""),
            created_at=data["created_at"],
            last_used_at=data.get("last_used_at"),
            revoked_at=data.get("revoked_at"),
        )
        for doc in docs
        if (data := doc.to_dict())
    ]
    # Sorted in memory — keeps the query single-field (auto-indexed), no
    # composite index to declare in firestore.indexes.json.
    keys.sort(key=lambda k: k.created_at, reverse=True)
    return keys


@router.delete("/{key_id}", status_code=204)
@limiter.limit("60/minute")
async def revoke_api_key(
    request: Request,
    key_id: str,
    db: DBSession,
    user: CurrentUser,
) -> None:
    """Revoke an API key. Idempotent — revoking a revoked key is a no-op.

    Revocation takes up to the auth cache TTL (~30s, see app.deps) to
    propagate to in-flight developer-API traffic.
    """
    doc_ref = db.collection("api_keys").document(key_id)
    doc = await doc_ref.get()
    data = doc.to_dict() if doc.exists else None
    # Same 404 for "doesn't exist" and "not yours" — don't leak key ids.
    if not data or data.get("owner_user_id") != str(user.id):
        raise NotFoundError("API key not found")

    if not data.get("revoked_at"):
        await doc_ref.set({"revoked_at": datetime.now(UTC)}, merge=True)
        logger.info("API key revoked: id=%s user=%s", key_id, user.id)
