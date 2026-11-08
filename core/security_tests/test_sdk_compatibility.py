"""SDK maintenance checks: only loopback HTTP, dummy keys, no database or .env."""

from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch


class SDKCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resources = ExitStack()
        cls.addClassCleanup(resources.close)
        cache = resources.enter_context(
            tempfile.TemporaryDirectory(prefix="autoflow-sdk-")
        )
        resources.enter_context(
            patch.dict(
                os.environ,
                {
                    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                    "OTEL_SDK_DISABLED": "true",
                    "DSPY_CACHEDIR": cache,
                    "OPENAI_API_KEY": "fixture-only-core-key",
                },
            )
        )
        for key in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]:
            resources.enter_context(patch.dict(os.environ, {key: ""}))
        connect, resolve = socket.socket.connect, socket.getaddrinfo

        def local_connect(sock, address):
            if not isinstance(address, tuple) or address[0] not in {"127.0.0.1", "::1"}:
                raise AssertionError("SDK test attempted a non-loopback connection")
            return connect(sock, address)

        def local_resolve(host, *args, **kwargs):
            if host not in {"127.0.0.1", "::1", "localhost", b"127.0.0.1"}:
                raise AssertionError("SDK test attempted non-loopback DNS")
            return resolve(host, *args, **kwargs)

        resources.enter_context(patch.object(socket.socket, "connect", local_connect))
        resources.enter_context(patch.object(socket, "getaddrinfo", local_resolve))
        cls.calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers["Content-Length"])
                if size > 65536:
                    self.send_error(413)
                    return
                payload = json.loads(self.rfile.read(size))
                cls.calls.append((self.path, self.headers.get("Authorization")))
                kind = "application/json"
                if self.path == "/v1/embeddings":
                    inputs = payload["input"]
                    count = (
                        len(inputs)
                        if isinstance(inputs, list)
                        and inputs
                        and not isinstance(inputs[0], int)
                        else 1
                    )
                    result = {
                        "object": "list",
                        "model": "text-embedding-3-small",
                        "data": [
                            {
                                "object": "embedding",
                                "index": i,
                                "embedding": [0.1, 0.2, 0.3],
                            }
                            for i in range(count)
                        ],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                    body = json.dumps(result).encode()
                elif self.path == "/v1/chat/completions":
                    content = (
                        json.dumps({"answer": 2})
                        if payload.get("response_format")
                        else "[[ ## answer ## ]]\n2\n[[ ## completed ## ]]"
                    )
                    base = {
                        "id": "chatcmpl-fixture",
                        "created": 1,
                        "model": "gpt-4o-mini",
                    }
                    if payload.get("stream"):
                        kind = "text/event-stream"
                        chunks = [
                            {
                                **base,
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant", "content": "2"},
                                        "finish_reason": None,
                                    }
                                ],
                            },
                            {
                                **base,
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ],
                            },
                        ]
                        body = (
                            "".join(
                                "data: " + json.dumps(chunk) + "\n\n"
                                for chunk in chunks
                            )
                            + "data: [DONE]\n\n"
                        ).encode()
                    else:
                        body = json.dumps(
                            {
                                **base,
                                "object": "chat.completion",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": content,
                                        },
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 1,
                                    "completion_tokens": 1,
                                    "total_tokens": 2,
                                },
                            }
                        ).encode()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        resources.callback(thread.join, 2)
        resources.callback(server.server_close)
        resources.callback(server.shutdown)
        cls.base = f"http://127.0.0.1:{server.server_port}/v1"

    def setUp(self):
        self.calls.clear()

    def llm(self):
        from autoflow.models.llms import LLM

        return LLM(
            model="openai/gpt-4o-mini",
            api_key="fixture-only-core-key",
            api_base=self.base,
            max_tokens=128,
            max_retries=0,
            additional_kwargs={"timeout": 2},
        )

    def assert_credentials(self, paths):
        self.assertEqual(
            self.calls, [(path, "Bearer fixture-only-core-key") for path in paths]
        )

    def test_chat_and_explicit_credentials(self):
        from llama_index.core.llms import ChatMessage

        result = self.llm().chat([ChatMessage(role="user", content="1+1")])
        self.assertIn("2", result.message.content)
        self.assert_credentials(["/v1/chat/completions"])

    def test_streaming_chat(self):
        from llama_index.core.llms import ChatMessage

        chunks = list(self.llm().stream_chat([ChatMessage(role="user", content="1+1")]))
        self.assertEqual("".join(chunk.delta or "" for chunk in chunks), "2")
        self.assert_credentials(["/v1/chat/completions"])

    def test_embedding_single_and_batch(self):
        from autoflow.models.embedding_models import EmbeddingModel

        model = EmbeddingModel(
            model_name="openai/text-embedding-3-small",
            dimensions=3,
            api_key="fixture-only-core-key",
            api_base=self.base,
            timeout=2,
        )
        self.assertEqual(model.get_query_embedding("one"), [0.1, 0.2, 0.3])
        self.assertEqual(
            model.get_text_embedding_batch(["one", "two"]), [[0.1, 0.2, 0.3]] * 2
        )
        self.assert_credentials(["/v1/embeddings", "/v1/embeddings"])

    def test_dspy_conversion_keeps_explicit_endpoint_and_key(self):
        import dspy
        from autoflow.models.llms.dspy import get_dspy_lm_by_llm

        lm = get_dspy_lm_by_llm(self.llm())
        lm.cache = False
        with dspy.context(lm=lm):
            result = dspy.Predict("question -> answer: float")(question="1+1")
        self.assertEqual(result.answer, 2)
        self.assert_credentials(["/v1/chat/completions"])
