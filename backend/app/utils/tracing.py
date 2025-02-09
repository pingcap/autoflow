from contextlib import contextmanager
from typing import Optional, Generator
from langfuse.client import StatefulSpanClient, StatefulClient
from langfuse.llama_index import LlamaIndexInstrumentor
from langfuse.llama_index._context import langfuse_instrumentor_context


class LangfuseContextManager:
    langfuse_client: Optional[StatefulSpanClient] = None

    def __init__(self, instrumentor: LlamaIndexInstrumentor):
        self.instrumentor = instrumentor

    @contextmanager
    def observe(self, **kwargs):
        try:
            self.instrumentor.start()
            with self.instrumentor.observe(**kwargs) as trace_client:
                trace_client.update(name=kwargs.get("trace_name"), **kwargs)
                self.langfuse_client = trace_client
                yield trace_client
        except Exception:
            raise
        finally:
            self.instrumentor.flush()
            self.instrumentor.stop()

    @contextmanager
    def span(
        self, parent_client: Optional[StatefulClient] = None, **kwargs
    ) -> Generator["StatefulSpanClient", None, None]:
        if parent_client:
            client = parent_client
        else:
            client = self.langfuse_client
        span = client.span(**kwargs)

        ctx = langfuse_instrumentor_context.get().copy()
        old_parent_observation_id = ctx.get("parent_observation_id")
        langfuse_instrumentor_context.get().update(
            {
                "parent_observation_id": span.id,
            }
        )

        try:
            yield span
        except Exception:
            raise
        finally:
            ctx.update(
                {
                    "parent_observation_id": old_parent_observation_id,
                }
            )
            langfuse_instrumentor_context.get().update(ctx)

    @property
    def trace_id(self) -> Optional[str]:
        if self.langfuse_client:
            return self.langfuse_client.trace_id
        else:
            return None

    @property
    def trace_url(self) -> Optional[str]:
        if self.langfuse_client:
            return self.langfuse_client.get_trace_url()
        else:
            return None
