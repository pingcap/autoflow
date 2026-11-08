"""Run transaction regressions against an owned, disposable loopback-only TiDB."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    backend = Path(__file__).resolve().parents[1]
    if (backend / ".env").exists():
        raise RuntimeError("Use an isolated checkout without a deployment .env file")
    # TiDB v8.5.4 multi-platform manifest (linux/amd64 and linux/arm64).
    image = os.environ.get(
        "SECURITY_TIDB_TEST_IMAGE",
        "pingcap/tidb@sha256:e986dadd59937fbfcbdfe7fd084421cc65988da8b795f75194b05a0292635ead",
    )
    if not re.fullmatch(r"pingcap/tidb(?::v[0-9.]+|@sha256:[a-f0-9]{64})", image):
        raise ValueError("Only an explicit pingcap/tidb release or digest is supported")
    container = command(
        "docker",
        "run",
        "-d",
        "--rm",
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--label",
        "codex.task=autoflow-security-tidb",
        "-p",
        "127.0.0.1::4000",
        "-p",
        "127.0.0.1::10080",
        image,
        "--store=unistore",
        "--path=/tmp/autoflow-security-test",
    )
    if not re.fullmatch(r"[a-f0-9]{64}", container):
        raise RuntimeError("Docker did not return an owned container ID")
    try:

        def port(number):
            value = command("docker", "port", container, f"{number}/tcp")
            match = re.fullmatch(r"127\.0\.0\.1:([0-9]+)", value)
            if not match:
                raise RuntimeError("Test container must bind only to loopback")
            return match[1]

        sql_port, status_port = port(4000), port(10080)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 45
        while True:
            try:
                with opener.open(
                    f"http://127.0.0.1:{status_port}/status", timeout=1
                ) as response:
                    version = json.loads(response.read(65536))["version"]
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned TiDB did not become ready") from None
                time.sleep(0.25)
        print(f"Owned loopback TiDB ready: {version}", flush=True)
        environment = dict(
            os.environ,
            SECURITY_TIDB_TEST_PORT=sql_port,
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        )
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--confcutdir=tests/tidb_security",
                "tests/tidb_security",
                "-q",
            ],
            cwd=backend,
            env=environment,
            timeout=180,
        ).returncode
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], check=False, stdout=subprocess.DEVNULL
        )


if __name__ == "__main__":
    raise SystemExit(main())
