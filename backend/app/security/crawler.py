"""Run each crawl in a fresh process with a hard deadline and no service credentials."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import psutil

from app.security.crawl_http import CrawlRejected, checked_url


def browser_cache_path():
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if configured:
        return configured
    if sys.platform == "darwin":
        return str(Path.home() / "Library/Caches/ms-playwright")
    return str(
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ms-playwright"
    )


def terminate_crawl(process):
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        for child in [parent, *descendants]:
            try:
                child.suspend()
            except psutil.Error:
                pass
        for child in reversed([parent, *descendants]):
            try:
                child.kill()
            except psutil.Error:
                pass
    except psutil.NoSuchProcess:
        pass
    process.wait()


def crawl(url: str, kind="page", private_origins=(), timeout=45):
    checked_url(url)
    payload = json.dumps(
        {"kind": kind, "url": url, "private_origins": list(private_origins)}
    ).encode()
    if len(payload) >= 16384:
        raise CrawlRejected("Crawl configuration is too large")
    with tempfile.TemporaryDirectory(prefix="autoflow-crawl-") as temporary:
        environment = {
            "PATH": os.defpath,
            "HOME": temporary,
            "TMPDIR": temporary,
            "LANG": "C.UTF-8",
            "PLAYWRIGHT_BROWSERS_PATH": browser_cache_path(),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "app.security.crawl_worker"],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(payload, timeout=timeout)
            if process.returncode != 0 or len(stdout) > 8 * 1024 * 1024:
                raise CrawlRejected("Crawler execution failed")
            result = json.loads(stdout)
            if "error" in result:
                raise CrawlRejected(result["error"])
            return result["result"]
        except subprocess.TimeoutExpired:
            terminate_crawl(process)
            raise CrawlRejected("Crawler execution deadline exceeded") from None
        finally:
            if process.poll() is None:
                terminate_crawl(process)
