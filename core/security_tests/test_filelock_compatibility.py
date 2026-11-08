"""Exercise Hugging Face cache locking against another local process."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FileLockCompatibility(unittest.TestCase):
    def test_cache_lock_excludes_another_process_and_can_be_reacquired(self):
        from huggingface_hub.utils import WeakFileLock

        attempt = """
import sys
from filelock import Timeout
from huggingface_hub.utils import WeakFileLock
try:
    with WeakFileLock(sys.argv[1], timeout=0.2):
        pass
except Timeout:
    sys.exit(0)
sys.exit(1)
"""
        with tempfile.TemporaryDirectory(prefix="autoflow-cache-lock-") as directory:
            path = Path(directory) / "model.lock"
            with WeakFileLock(path, timeout=1):
                result = subprocess.run(
                    [sys.executable, "-c", attempt, str(path)],
                    env={**os.environ, "HF_HUB_OFFLINE": "1"},
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())
            with WeakFileLock(path, timeout=1):
                self.assertTrue(path.exists())
