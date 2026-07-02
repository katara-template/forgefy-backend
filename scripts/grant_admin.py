"""Grant (or revoke) admin privileges for a user by email.

Usage:
    python scripts/grant_admin.py user@example.com
    python scripts/grant_admin.py user@example.com --revoke
"""
import argparse
import asyncio
import sys

from app.db.firebase import get_firestore_client


async def set_admin(email: str, is_admin: bool) -> None:
    db = get_firestore_client()
    query = db.collection("users").where("email", "==", email).limit(1)
    docs = [doc async for doc in query.stream()]
    if not docs:
        print(f"No user found with email {email!r}", file=sys.stderr)
        raise SystemExit(1)

    doc = docs[0]
    await doc.reference.set({"is_admin": is_admin}, merge=True)
    print(f"{email} is_admin={is_admin}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    parser.add_argument("--revoke", action="store_true")
    args = parser.parse_args()
    asyncio.run(set_admin(args.email, is_admin=not args.revoke))
