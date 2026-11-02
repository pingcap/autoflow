import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def test_regression_workflow_passes_hostile_inputs_as_literal_arguments(tmp_path):
    workflow = yaml.safe_load(
        (Path(__file__).parents[3] / ".github/workflows/regression.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["build"]["steps"]
        if step["name"] == "Run tests"
    )
    marker = tmp_path / "unexpected-shell-command"
    dataset = f'$(touch "{marker}"); " dataset\nwith newline'
    chat_engine = f'engine; touch "{marker}" #'
    # Model Actions expression expansion as well as actual shell execution. A
    # future regression to direct interpolation must fail this test.
    script = step["run"]
    for name, value in {
        "dataset": dataset,
        "tidbAIChatEngine": chat_engine,
        "llmProvider": "openai",
    }.items():
        script = script.replace("${{ inputs." + name + " }}", value)
    (tmp_path / "backend").mkdir()
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    capture = tmp_path / "arguments.json"
    uv = bin_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nwith open(os.environ['ARG_CAPTURE'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    uv.chmod(0o700)
    environment = os.environ | {
        "PATH": str(bin_path) + os.pathsep + os.environ["PATH"],
        "ARG_CAPTURE": str(capture),
        "REGRESSION_DATASET": dataset,
        "REGRESSION_CHAT_ENGINE": chat_engine,
        "REGRESSION_LLM_PROVIDER": "openai",
    }
    subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    arguments = json.loads(capture.read_text())
    assert arguments[arguments.index("--dataset") + 1] == dataset
    assert arguments[arguments.index("--tidb-ai-chat-engine") + 1] == chat_engine
    assert not marker.exists()
