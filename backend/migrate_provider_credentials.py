"""Validate/migrate provider credential blobs without logging their contents.

Dry-run is the default. Stop API/background workers and back up the database first.
The caller supplies the current DATA_ENCRYPTION_KEY; legacy keys are read only
from private files. All writes commit together or roll back together.
"""

import argparse
import base64
import json
from pathlib import Path
import stat

from sqlalchemy import text

from app.utils.aes import AESCipher, MAGIC
from app.utils.legacy_cipher import decrypt_legacy_credentials

TABLES = ("llms", "embedding_models", "reranker_models")


def read_private_file(path: Path) -> bytes:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_mode & 0o077
        or info.st_size > 64 * 1024
    ):
        raise ValueError("Key files must be private regular files (chmod 600)")
    return path.read_bytes().rstrip(b"\r\n")


def migrate_connection(
    connection, current_key, *, legacy_secret=None, previous_key=None, apply=False
):
    counts = {"checked": 0, "needs_migration": 0, "migrated": 0}
    for table in TABLES:
        current = AESCipher(current_key, context=f"{table}.credentials")
        previous = (
            AESCipher(previous_key, context=f"{table}.credentials")
            if previous_key
            else None
        )
        last_id = 0
        lock = " FOR UPDATE" if apply and connection.dialect.name != "sqlite" else ""
        while True:
            rows = (
                connection.execute(
                    text(
                        f"SELECT id, credentials FROM {table} WHERE id > :last_id AND credentials IS NOT NULL ORDER BY id LIMIT 100{lock}"
                    ),
                    {"last_id": last_id},
                )
                .mappings()
                .all()
            )
            if not rows:
                break
            for row in rows:
                last_id = row["id"]
                encrypted = bytes(row["credentials"])
                needs_migration = True
                if encrypted.startswith(MAGIC):
                    try:
                        plain = current.decrypt(encrypted)
                        needs_migration = False
                    except ValueError:
                        if previous is None:
                            raise
                        plain = previous.decrypt(encrypted)
                else:
                    if legacy_secret is None:
                        raise ValueError(
                            "A private legacy signing-key file is required"
                        )
                    plain = decrypt_legacy_credentials(encrypted, legacy_secret)
                # CFB has no authenticity guarantee. Validate the JSON structure,
                # but do not claim this can prove historical ciphertext integrity.
                value = json.loads(plain)
                if not isinstance(value, (str, dict, list, type(None))):
                    raise ValueError("Unexpected credential structure")
                counts["checked"] += 1
                if needs_migration:
                    replacement = current.encrypt(plain)
                    counts["needs_migration"] += 1
                    if apply:
                        result = connection.execute(
                            text(
                                f"UPDATE {table} SET credentials = :replacement WHERE id = :id AND credentials = :original"
                            ),
                            {
                                "replacement": replacement,
                                "id": row["id"],
                                "original": encrypted,
                            },
                        )
                        if result.rowcount != 1:
                            raise ValueError(
                                "Credentials changed concurrently; stop workers and retry"
                            )
                        counts["migrated"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit all changes after validation; default is dry-run",
    )
    legacy = parser.add_mutually_exclusive_group()
    legacy.add_argument("--legacy-signing-key-file", type=Path)
    legacy.add_argument(
        "--legacy-env-file",
        type=Path,
        help="Private backup of the old .env, containing SECRET_KEY",
    )
    parser.add_argument(
        "--previous-data-key-file",
        type=Path,
        help="Private file containing the previous base64 DATA_ENCRYPTION_KEY",
    )
    args = parser.parse_args()
    try:
        legacy_secret = None
        if args.legacy_signing_key_file:
            legacy_secret = read_private_file(args.legacy_signing_key_file)
        elif args.legacy_env_file:
            from dotenv import dotenv_values
            from io import StringIO

            values = dotenv_values(
                stream=StringIO(read_private_file(args.legacy_env_file).decode()),
                interpolate=False,
            )
            legacy_secret = (values.get("SECRET_KEY") or "").encode()
            if len(legacy_secret) < 32:
                raise ValueError(
                    "The legacy environment backup lacks a valid signing key"
                )
        previous = (
            base64.b64decode(
                read_private_file(args.previous_data_key_file), validate=True
            )
            if args.previous_data_key_file
            else None
        )
        from app.core.config import settings
        from app.core.db import engine

        current = base64.b64decode(
            settings.DATA_ENCRYPTION_KEY.get_secret_value(), validate=True
        )
        with engine.begin() as connection:
            counts = migrate_connection(
                connection,
                current,
                legacy_secret=legacy_secret,
                previous_key=previous,
                apply=args.apply,
            )
    except Exception as error:
        # Database/decoder errors may contain parameters or plaintext fragments.
        print(
            f"Credential migration did not complete ({type(error).__name__}); verify database state before retrying."
        )
        raise SystemExit(1) from None
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", **counts}))


if __name__ == "__main__":
    main()
