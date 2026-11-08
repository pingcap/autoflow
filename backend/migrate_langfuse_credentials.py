"""Encrypt legacy Langfuse settings, or rotate their data-encryption key.

Stop API/worker writers and take a private database backup before --apply.
The default dry-run performs reads and in-memory validation only.
"""

import argparse
import base64
import json
from pathlib import Path

from pydantic import SecretStr
from sqlmodel import Session

from app.security.langfuse_settings import (
    LangfuseUpdate,
    _values,
    snapshot,
    decrypt_secret,
    encrypt_secret,
    save_configuration,
    validate_host,
)
from migrate_provider_credentials import read_private_file


def migrate_session(session, current_key, *, previous_key=None, apply=False):
    values = _values(session, lock=apply)
    record = snapshot(values)
    needs_migration = False
    if record["version"] == 0:
        secret = record["legacy_secret"]
        needs_migration = bool(secret)
    else:
        try:
            secret = decrypt_secret(record, key=current_key)
        except ValueError:
            if previous_key is None:
                raise
            secret = decrypt_secret(record, key=previous_key)
            needs_migration = True
        needs_migration |= bool(values.get("langfuse_secret_key"))
    if needs_migration:
        host = validate_host(record["host"])
        if secret and not record["public_key"]:
            raise ValueError("Langfuse configuration lacks a public key")
        # Validate encryption before any write, including in dry-run mode.
        encrypted = encrypt_secret(secret, host, record["public_key"], key=current_key)
        if (
            decrypt_secret(
                {**record, "host": host, "secret_ciphertext": encrypted},
                key=current_key,
            )
            != secret
        ):
            raise ValueError("Langfuse credential validation failed")
        if apply:
            save_configuration(
                session,
                LangfuseUpdate(
                    host=host,
                    public_key=record["public_key"],
                    revision=record["revision"],
                    secret_key=SecretStr(secret) if secret else None,
                    clear_secret=not bool(secret),
                ),
                key=current_key,
            )
    return {
        "needs_migration": int(needs_migration),
        "migrated": int(needs_migration and apply),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--previous-data-key-file", type=Path)
    args = parser.parse_args()
    try:
        from app.core.db import engine
        from app.models.base import get_aes_key

        previous = (
            base64.b64decode(
                read_private_file(args.previous_data_key_file), validate=True
            )
            if args.previous_data_key_file
            else None
        )
        with Session(engine) as session, session.begin():
            counts = migrate_session(
                session, get_aes_key(), previous_key=previous, apply=args.apply
            )
    except Exception as error:
        print(
            f"Langfuse migration did not complete ({type(error).__name__}); no credential values are reported."
        )
        raise SystemExit(1) from None
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", **counts}))


if __name__ == "__main__":
    main()
