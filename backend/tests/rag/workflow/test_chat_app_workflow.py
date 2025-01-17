import unittest

import pytest
from sqlmodel import Session


from app.core.db import engine
from app.rag.chat_config import ChatEngineConfig
from app.rag.workflows.chat_app.ChatAppWorkflow import AppChatFlow


@pytest.mark.asyncio
async def test_something():
    with Session(engine) as db_session:
        engine_config = ChatEngineConfig.load_from_db(db_session, "default")
        flow = AppChatFlow(db_session, "test", engine_config)
        result = await flow.run()
        print(result)


@pytest.mark.asyncio
async def test_something():
    with Session(engine) as db_session:
        engine_config = ChatEngineConfig.load_from_db(db_session, "default")
        flow = AppChatFlow(db_session, "test", engine_config)
        result = await flow.run()
        print(result)


if __name__ == "__main__":
    unittest.main()
