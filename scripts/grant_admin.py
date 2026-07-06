"""Grant (or revoke) admin privileges for a user by email.

If no account exists with that email yet, one is created (prompting for a
password unless --password is given) with the requested admin status.

Usage (run as a module from the project root, e.g. c:\\...\\forgefy-backend>):
    python -m scripts.grant_admin user@example.com
    python -m scripts.grant_admin user@example.com --revoke
    python -m scripts.grant_admin newadmin@example.com --password "SomeStrongPass123"
"""
import argparse
import asyncio
import getpass
import sys
import uuid
from datetime import UTC, datetime

from app.core.security import hash_password
from app.db.firebase import get_firestore_client


async def set_admin(email: str, is_admin: bool, password: str | None) -> None:
    db = get_firestore_client()
    query = db.collection("users").where("email", "==", email).limit(1)
    docs = [doc async for doc in query.stream()]

    if docs:
        if password:
            print(
                f"Note: {email!r} already exists — ignoring --password (use the normal "
                "login/change-password flow to update it).",
                file=sys.stderr,
            )
        await docs[0].reference.set({"is_admin": is_admin}, merge=True)
        print(f"{email} is_admin={is_admin}")
        return

    if not password:
        password = getpass.getpass(f"No account found for {email!r} — set a password to create one: ")
        if not password:
            print("A password is required to create a new account.", file=sys.stderr)
            raise SystemExit(1)

    user_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    await db.collection("users").document(user_id).set({
        "email": email,
        "hashed_password": hash_password(password),
        "tier": "free",
        "is_admin": is_admin,
        "created_at": now,
        "updated_at": now,
    })
    print(f"Created {email} (id={user_id}) is_admin={is_admin}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    parser.add_argument("--revoke", action="store_true", help="Set is_admin=False instead of True")
    parser.add_argument(
        "--password",
        default=None,
        help="Password to use if the account needs to be created. Prompted securely if omitted.",
    )
    args = parser.parse_args()
    asyncio.run(set_admin(args.email, is_admin=not args.revoke, password=args.password))
