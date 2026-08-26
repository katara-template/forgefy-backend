"""One-time migration: import custom email/password users into Firebase Auth.

Every Firestore users/{id} doc with a bcrypt hashed_password is imported into
Firebase Authentication (Firebase natively supports bcrypt password-hash
import, so existing passwords keep working), using the Firestore doc id as
the Firebase UID. The doc is then stamped with firebase_uid so the
/api/v1/auth/oauth endpoint binds sign-ins to the right account.

OAuth-only accounts (empty hashed_password) are skipped — they get their
firebase_uid backfilled automatically on their next sign-in.

Usage (from the forgefy-backend project root, with Firebase credentials in
FIREBASE_CREDENTIALS_JSON or firebase-credentials.json):
    python -m scripts.migrate_users_to_firebase --dry-run
    python -m scripts.migrate_users_to_firebase
"""
import argparse
import asyncio
import sys

from firebase_admin import auth as fb_auth

from app.db.firebase import get_firestore_client

_IMPORT_BATCH = 1000  # Firebase import_users hard limit per call


async def migrate(dry_run: bool) -> None:
    db = get_firestore_client()

    to_import: list[fb_auth.ImportUserRecord] = []
    to_stamp: list = []  # doc references to write firebase_uid onto
    skipped_oauth = 0
    already_done = 0

    async for doc in db.collection("users").stream():
        data = doc.to_dict() or {}
        if data.get("firebase_uid"):
            already_done += 1
            continue
        hashed = data.get("hashed_password") or ""
        if not hashed.startswith("$2"):  # not a bcrypt hash → OAuth-only account
            skipped_oauth += 1
            continue
        to_import.append(
            fb_auth.ImportUserRecord(
                uid=doc.id,
                email=data["email"],
                password_hash=hashed.encode("utf-8"),
            )
        )
        to_stamp.append(doc.reference)

    print(
        f"Found {len(to_import)} password user(s) to import "
        f"({already_done} already migrated, {skipped_oauth} OAuth-only skipped)."
    )
    if dry_run or not to_import:
        return

    failed_uids: set[str] = set()
    for i in range(0, len(to_import), _IMPORT_BATCH):
        batch = to_import[i : i + _IMPORT_BATCH]
        result = fb_auth.import_users(batch, hash_alg=fb_auth.UserImportHash.bcrypt())
        for err in result.errors:
            failed = batch[err.index]
            failed_uids.add(failed.uid)
            print(f"  FAILED {failed.email} (uid={failed.uid}): {err.reason}", file=sys.stderr)
        print(f"Imported {result.success_count}/{len(batch)} (batch {i // _IMPORT_BATCH + 1})")

    # Stamp firebase_uid (== doc id) so the backend binds sign-ins to the doc.
    stamped = 0
    for ref in to_stamp:
        if ref.id in failed_uids:
            continue
        await ref.set({"firebase_uid": ref.id}, merge=True)
        stamped += 1
    print(f"Stamped firebase_uid on {stamped} user doc(s). {len(failed_uids)} import error(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report what would be migrated, change nothing")
    args = parser.parse_args()
    asyncio.run(migrate(dry_run=args.dry_run))
