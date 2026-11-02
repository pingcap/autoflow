import base64
import json
import os

import pytest
from cryptography.hazmat.decrepit.ciphers.modes import CFB
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from sqlalchemy import create_engine, text

from app.core.config import Settings
from app.utils.aes import AESCipher, MAGIC
from migrate_provider_credentials import migrate_connection, read_private_file


def legacy_encrypt(value, secret):
    padder = padding.PKCS7(128).padder()
    padded = padder.update(json.dumps(value).encode()) + padder.finalize()
    iv = os.urandom(16)
    encryptor = Cipher(algorithms.AES(secret[:32]), CFB(iv)).encryptor()
    return iv + encryptor.update(padded) + encryptor.finalize()


def test_credentials_authenticate_ciphertext_nonce_key_and_column_context():
    key = os.urandom(32)
    cipher = AESCipher(key, context="llms.credentials")
    encrypted = cipher.encrypt('"private-provider-token"')
    assert encrypted.startswith(MAGIC)
    assert b"private-provider-token" not in encrypted
    assert encrypted != cipher.encrypt('"private-provider-token"')
    assert cipher.decrypt(encrypted) == '"private-provider-token"'
    for offset in [len(MAGIC), len(encrypted) - 1]:
        altered = bytearray(encrypted)
        altered[offset] ^= 1
        with pytest.raises(ValueError, match="authentication"):
            cipher.decrypt(altered)
    for other in [
        AESCipher(os.urandom(32), context="llms.credentials"),
        AESCipher(key, context="embedding_models.credentials"),
    ]:
        with pytest.raises(ValueError, match="authentication"):
            other.decrypt(encrypted)
    with pytest.raises(ValueError, match="Legacy"):
        cipher.decrypt(legacy_encrypt("token", os.urandom(48)))


def fixture_database(blobs):
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        for table in ("llms", "embedding_models", "reranker_models"):
            connection.execute(
                text(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, credentials BLOB)")
            )
        for index, blob in enumerate(blobs, 1):
            connection.execute(
                text("INSERT INTO llms VALUES (:id, :data)"),
                {"id": index, "data": blob},
            )
    return engine


def test_explicit_legacy_migration_has_dry_run_atomic_commit_and_idempotence():
    old, new = os.urandom(48), os.urandom(32)
    value = {"private_key": "fixture-only", "unicode": "测试"}
    blob = legacy_encrypt(value, old)
    engine = fixture_database([blob])
    with engine.begin() as connection:
        counts = migrate_connection(connection, new, legacy_secret=old)
        assert counts == {"checked": 1, "needs_migration": 1, "migrated": 0}
        assert (
            connection.execute(text("SELECT credentials FROM llms")).scalar_one()
            == blob
        )
    with engine.begin() as connection:
        assert (
            migrate_connection(connection, new, legacy_secret=old, apply=True)[
                "migrated"
            ]
            == 1
        )
    with engine.begin() as connection:
        migrated = connection.execute(text("SELECT credentials FROM llms")).scalar_one()
        assert (
            json.loads(AESCipher(new, context="llms.credentials").decrypt(migrated))
            == value
        )
        assert migrate_connection(connection, new, apply=True)["needs_migration"] == 0
    engine.dispose()


def test_one_invalid_row_rolls_back_all_previous_secret_updates():
    old, new = os.urandom(48), os.urandom(32)
    blob = legacy_encrypt("valid", old)
    engine = fixture_database([blob, b"corrupt"])
    with pytest.raises(ValueError):
        with engine.begin() as connection:
            migrate_connection(connection, new, legacy_secret=old, apply=True)
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT credentials FROM llms WHERE id=1")
            ).scalar_one()
            == blob
        )
    engine.dispose()


def test_authenticated_key_rotation_requires_the_previous_key():
    old, new = os.urandom(32), os.urandom(32)
    value = '"fixture-token"'
    blob = AESCipher(old, context="llms.credentials").encrypt(value)
    engine = fixture_database([blob])
    with pytest.raises(ValueError):
        with engine.begin() as connection:
            migrate_connection(connection, new, apply=True)
    with engine.begin() as connection:
        assert (
            migrate_connection(connection, new, previous_key=old, apply=True)[
                "migrated"
            ]
            == 1
        )
    with engine.connect() as connection:
        changed = connection.execute(text("SELECT credentials FROM llms")).scalar_one()
        assert AESCipher(new, context="llms.credentials").decrypt(changed) == value
    engine.dispose()


def test_configuration_representations_hide_signing_provider_and_database_secrets():
    signing = "fixture-signing-secret-" + os.urandom(32).hex()
    encryption = base64.b64encode(os.urandom(32)).decode()
    database_password = "fixture-database-password"
    provider = "fixture-provider-secret"
    config = Settings(
        _env_file=None,
        SECRET_KEY=signing,
        DATA_ENCRYPTION_KEY=encryption,
        TIDB_DATABASE="test",
        TIDB_PASSWORD=database_password,
        EVALUATION_OPENAI_API_KEY=provider,
    )
    serialized = repr(config) + config.model_dump_json()
    for secret in [signing, encryption, database_password, provider]:
        assert secret not in serialized
    assert database_password in str(config.SQLALCHEMY_DATABASE_URI)


def test_private_key_files_reject_world_readability_and_symlinks(tmp_path):
    path = tmp_path / "secret"
    path.write_bytes(b"fixture")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private regular"):
        read_private_file(path)
    path.chmod(0o600)
    assert read_private_file(path) == b"fixture"
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="private regular"):
        read_private_file(link)
