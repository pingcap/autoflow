"""Exercise the real sidecar container with synthetic inputs and no external credentials."""

import json
import os
import secrets
import subprocess
import time
import urllib.error
import urllib.request

image = "autoflow-sidecar-security-test"
name = "autoflow-sidecar-security-test"
environment = {**os.environ, "SIDECAR_AUTH_TOKEN": secrets.token_urlsafe(32)}
subprocess.run(
    [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--publish",
        "127.0.0.1::5001",
        "--memory",
        "8g",
        "--cpus",
        "2",
        "--env",
        "SIDECAR_AUTH_TOKEN",
        "--env",
        "PRE_LOAD_DEFAULT_EMBEDDING_MODEL=false",
        image,
    ],
    env=environment,
    check=True,
    stdout=subprocess.DEVNULL,
)
try:
    port = (
        subprocess.check_output(["docker", "port", name, "5001/tcp"], text=True)
        .strip()
        .rsplit(":", 1)[1]
    )
    base = "http://127.0.0.1:" + port
    for attempt in range(60):
        try:
            with urllib.request.urlopen(
                base + "/api/v1/healthz", timeout=2
            ) as response:
                assert response.status == 200
            break
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    else:
        raise RuntimeError("Sidecar did not become healthy")

    def post(payload, authorized):
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = "Bearer " + environment["SIDECAR_AUTH_TOKEN"]
        request = urllib.request.Request(
            base + "/api/v1/embedding",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    assert post({"sentences": ["hello"]}, False) == 401
    assert post({"sentences": ["hello"], "model": "unapproved/model"}, True) == 422
    assert post({"sentences": ["hello"] * 33}, True) == 422
    assert (
        subprocess.check_output(["docker", "exec", name, "id", "-u"], text=True).strip()
        == "10001"
    )
    subprocess.run(
        [
            "docker",
            "exec",
            name,
            "python",
            "-c",
            "from sentence_transformers import SentenceTransformer, CrossEncoder; import inspect; assert 'local_files_only' in inspect.signature(CrossEncoder).parameters; assert 'revision' in inspect.signature(SentenceTransformer).parameters",
        ],
        check=True,
    )

    def infer(endpoint, payload):
        request = urllib.request.Request(
            base + endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + environment["SIDECAR_AUTH_TOKEN"],
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            assert response.status == 200
            return json.load(response)

    embedded = infer(
        "/api/v1/embedding", {"sentences": ["TiDB is a distributed SQL database."]}
    )
    assert len(embedded["embeddings"]) == 1 and len(embedded["embeddings"][0]) == 1024
    ranked = infer(
        "/api/v1/reranker",
        {
            "query": "What is TiDB?",
            "passages": ["TiDB is a distributed SQL database.", "A banana is a fruit."],
        },
    )
    assert len(ranked["scores"]) == 2 and ranked["scores"][0] > ranked["scores"][1]
    print(
        "Sidecar authentication, bounds, non-root runtime and both pinned model inference checks passed."
    )
finally:
    subprocess.run(["docker", "stop", name], check=False, stdout=subprocess.DEVNULL)
