import pytest
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    Event,
    StartEvent,
    StopEvent,
)
from openai import OpenAI


class DoSomethingEvent(Event):
    """Do something"""

    pass


class TestLlamaindexFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> DoSomethingEvent:
        db_session = ev.get("db_session")
        openai = OpenAI(api_key="xxx")
        await ctx.set("openai", openai)
        return DoSomethingEvent()

    @step
    async def doing(self, ctx: Context, ev: DoSomethingEvent) -> StopEvent:
        openai: OpenAI = await ctx.get("openai")

        print(f"Hello world {openai.api_key}")
        return StopEvent()


@pytest.mark.asyncio
async def test_llamaindex_workflow():
    flow = TestLlamaindexFlow()
    await flow.run()
