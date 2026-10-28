import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def test_queue_lifecycle_without_host_probes(self):
        with tempfile.TemporaryDirectory(prefix="gsched-cli-") as tmp:
            root = Path(tmp)
            config = root / "config" / "hosts.toml"
            env = {**os.environ, "GSCHED_CONFIG": str(config),
                   "GSCHED_STATE": str(root / "state"), "GSCHED_WORK": str(root / "work")}

            def cli(*args, success=True):
                result = subprocess.run([sys.executable, "-m", "gsched", *args],
                                        cwd=root, env=env, text=True, capture_output=True, timeout=15)
                if success:
                    self.assertEqual(result.returncode, 0, f"{args}: {result.stdout}\n{result.stderr}")
                else:
                    self.assertNotEqual(result.returncode, 0, f"{args} unexpectedly succeeded")
                return result

            cli("init")
            self.assertTrue(config.is_file())
            original = config.read_bytes()
            self.assertEqual(cli("submit", "-n", "x", "--", "echo", "hi").stdout.strip(), "1")
            listing = cli("ls").stdout
            self.assertIn("queued", listing)
            self.assertRegex(listing, r"(?m)^\s*1\s+queued\s+")
            self.assertIn("x", listing)
            job = json.loads(cli("show", "1").stdout)
            self.assertEqual((job["id"], job["name"], job["state"]), (1, "x", "queued"))
            self.assertEqual(job["spec"]["cmd"], "echo hi")
            self.assertEqual(job["spec"]["workdir"], str(root))
            self.assertIsNone(job["host"])
            self.assertIn("cancelled", cli("cancel", "1").stdout)
            job = json.loads(cli("show", "1").stdout)
            self.assertEqual(job["state"], "cancelled")
            self.assertIsNotNone(job["ended"])
            self.assertIn("queued again", cli("retry", "1").stdout)
            job = json.loads(cli("show", "1").stdout)
            self.assertEqual(job["state"], "queued")
            self.assertEqual(job["note"], "retried")
            self.assertIsNone(job["ended"])
            self.assertIsNone(job["host"])
            self.assertEqual(job["attempts"], 0)
            self.assertIn("--force", cli("init", success=False).stderr)
            self.assertEqual(config.read_bytes(), original)
            cli("init", "--force")
            self.assertEqual(config.read_bytes(), original)
            self.assertTrue((root / "state" / "gsched.db").is_file())
            self.assertFalse((root / "work").exists())
