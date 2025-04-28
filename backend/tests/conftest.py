import pytest
from dotenv import load_dotenv

from app.auth.api_keys import ApiKeyManager
from app.core.db import engine
from app.models.auth import User
from sqlmodel import Session, select


@pytest.fixture(scope="session", autouse=True)
def env():
    print("Loading environment variables")
    load_dotenv()


@pytest.fixture(scope="session", autouse=True)
def test_api_key():
    api_key_manager = ApiKeyManager()
    with Session(engine) as session:
        user = (
            session.exec(select(User).where(User.email == "admin@example.com"))
        ).one()
        _, raw_api_key = api_key_manager.create_api_key(
            session, user, "Created by unit test"
        )
        return raw_api_key
