import asyncio
from uuid import uuid4

import pytest
from sqlmodel import Session

from app.models import User, ChatEngine, ApiKey, ChatVisibility
from tests.security.test_answer_cache import OWNER, cache_answer, current, lookups
from tests.security.test_session_revocation import scenario


@pytest.mark.parametrize("rollback", [False, True])
def test_tidb_password_revocation_and_late_issuance_are_atomic(database, rollback):
    url = database.url.set(drivername="mysql+asyncmy").render_as_string(
        hide_password=False
    )
    asyncio.run(scenario(fail_revoke=rollback, database_url=url))


def test_tidb_answer_cache_json_filters_preserve_each_principal_boundary(database):
    other = uuid4()
    with Session(database, expire_on_commit=False) as session:
        session.add(
            User(id=OWNER, email="owner@example.test", hashed_password="fixture")
        )
        session.add(
            User(id=other, email="other@example.test", hashed_password="fixture")
        )
        session.add(ChatEngine(id=1, name="fixture-one"))
        session.add(ChatEngine(id=2, name="fixture-two"))
        session.commit()
        for key in [8, 9]:
            session.add(
                ApiKey(
                    id=key,
                    user_id=OWNER,
                    description="fixture",
                    api_key_display="fixture",
                    hashed_secret=f"fixture-{key}",
                )
            )
        session.commit()
        own = cache_answer(session)
        assert lookups(session, current()) == [[own], [own]]
        for changes in [
            {"user_id": other},
            {"engine_id": 2},
            {"visibility": ChatVisibility.PUBLIC},
            {"user_id": None, "browser_id": "guest-a"},
            {"api_key_id": 9},
        ]:
            cache_answer(session, chat_changes=changes)
        assert lookups(session, current()) == [[own], [own]]
        machine = cache_answer(session, chat_changes={"api_key_id": 8})
        assert lookups(session, current(api_key_id=8)) == [[machine], [machine]]
        guest = cache_answer(
            session, chat_changes={"user_id": None, "browser_id": "guest-b"}
        )
        assert lookups(session, current(user_id=None, browser_id="guest-b")) == [
            [guest],
            [guest],
        ]
        assert lookups(session, current(user_id=None, browser_id="unrelated")) == [
            [],
            [],
        ]
