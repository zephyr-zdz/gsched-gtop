import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from support import csched


def host(name="node1", **overrides):
    return {"name": name, "backend": "node", "ssh": "local", "pref": 1, "enabled": True,
            "max_cpus": 8, "reserve_cpus": 0, "mem_reserve_gb": 0, "quiet_hours": "",
            "home": "/home/me", "jobroot": "/home/me/jobs", "setup": "", "default_time": "1-00:00:00",
            "local_disk": "", "node_frac": 1.0, "cores_per_free_gpu": 0, **overrides}


def node_view(**overrides):
    return {"ok": True, "ncpu": 8.0, "load": 0.0, "mem_avail_gb": 64.0, "ours_cpus": 0,
            "ours_mem_gb": 0, "nodes": {}, "squeue": [], "shards": {}, **overrides}


def slurm_node(**overrides):
    return {"up": True, "state": "idle", "cpu_idle": 20, "cpu_total": 32, "mem_total_gb": 128,
            "mem_unalloc_gb": 64, "gpus": 4, "gpus_used": 1, **overrides}


def slurm_view(**overrides):
    return {"ok": True, "nodes": {"n01": slurm_node()}, "squeue": [], "ours_cpus": 0,
            "ours_pending": 0, "ours_node": {}, "shards": {}, **overrides}


class ConfigAndPathTests(unittest.TestCase):
    def test_load_config_defaults_and_overrides(self):
        text = """
[defaults]
poll_seconds = 12
[hosts.node1]
ssh = "local"
max_cpus = 3
[hosts.cluster]
backend = "slurm"
partition = "cpu"
"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "csched.toml"
            cfg.write_text(text)
            with patch.object(csched, "CONFIG", cfg):
                d, hosts = csched.load_config()
        self.assertEqual((d["poll_seconds"], d["mem_per_cpu_gb"], d["max_batch"], d["retries"]), (12, 2, 8, 2))
        self.assertEqual(hosts["node1"]["pref"], 9)
        self.assertEqual(hosts["node1"]["max_cpus"], 3)
        self.assertEqual(hosts["node1"]["jobroot"], csched.LOCAL_HOME + "/scratch/csched/jobs")
        self.assertEqual(hosts["cluster"]["partition"], "cpu")
        self.assertEqual(hosts["cluster"]["node_frac"], 1.0)

    def test_remap_only_under_local_home(self):
        h = host(home="/remote/me")
        with patch.object(csched, "LOCAL_HOME", "/local/me"):
            self.assertEqual(csched.remap("/local/me/work", h), "/remote/me/work")
            self.assertEqual(csched.remap("/local/me", h), "/remote/me")
            self.assertEqual(csched.remap("/local/me2/work", h), "/local/me2/work")
            self.assertEqual(csched.remap(Path("/data/x"), h), "/data/x")


class PlacementMathTests(unittest.TestCase):
    def test_node_slots_reserve_load_and_planned(self):
        h = host(max_cpus=8, reserve_cpus=2, mem_reserve_gb=8)
        spec = {"cpus": 2, "mem_gb": 4}
        self.assertEqual(csched.slots(spec, h, node_view(), {}), (3, {}))
        self.assertEqual(csched.slots(spec, h, node_view(load=2.0), {})[0], 2)
        self.assertEqual(csched.slots(spec, h, node_view(), {"cpus": 2})[0], 2)
        self.assertEqual(csched.slots(spec, h, node_view(mem_avail_gb=15), {})[0], 1)
        self.assertEqual(csched.slots(spec, h, node_view(ok=False), {})[0], 0)
        h["quiet_hours"] = "00:00-23:59"
        self.assertEqual(csched.slots(spec, h, node_view(), {})[0], 0)

    def test_slurm_node_room_cap_and_gpu_keepback(self):
        h = host("cluster", backend="slurm", max_cpus=16, node_frac=0.5, cores_per_free_gpu=2)
        v = slurm_view()
        self.assertEqual(csched.node_cap(h, v["nodes"]["n01"]), 16)
        self.assertEqual(csched.node_room(h, v, "n01"), 14)  # idle 20 - 2 × 3 free GPUs
        v["ours_node"] = {"n01": 4}
        self.assertEqual(csched.node_room(h, v, "n01"), 12)
        spec = {"cpus": 4, "mem_gb": 8}
        self.assertEqual(csched.slots(spec, h, v, {}), (3, {"n01": 3}))

    def test_plan_slices_tasks_into_lanes_and_rounds_mem(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(csched, "STATE", Path(tmp)):
                c = csched.db()
                try:
                    spec = {"cpus": 2, "mem_gb": 4, "ntasks": 4}
                    jid = c.execute("insert into jobs(name, spec, state, submitted, ntasks) values(?,?,?,?,?)",
                                    ("sweep", json.dumps(spec), "queued", 0, 4)).lastrowid
                    c.executemany("insert into tasks(job, idx, state) values(?,?,'queued')", [(jid, i) for i in range(4)])
                    c.commit()
                    d = {"max_batch": 8}
                    out = csched.plan(c, d, {"node1": host()}, {"node1": node_view()})
                finally:
                    c.close()
        self.assertEqual(len(out), 1)
        j, name, idxs, lanes, node = out[0]
        self.assertEqual((j["id"], name, idxs, lanes, node), (jid, "node1", [0, 1, 2, 3], 4, None))


class ProbeAndScriptTests(unittest.TestCase):
    def test_probe_node_fields_and_shards(self):
        out = "ncpu 8\nload 0.25\nmemavail 4194304\nshard 3 2 alive\n"
        with patch.object(csched, "sh", lambda *a, **k: (0, out, "")):
            v = csched.probe(host(), [{"id": 3, "jobdir": "/jobs/3"}])
        self.assertTrue(v["ok"])
        self.assertEqual((v["ncpu"], v["load"], v["mem_avail_gb"]), (8.0, 0.25, 4.0))
        self.assertEqual(v["shards"][3], {"rc": 2, "proc": "alive"})

    def test_probe_slurm_sinfo_and_squeue(self):
        out = "\n".join(["#sinfo",
                         "n01 idle 0/8/0/8 32768 0 gpu:2 gpu:0",
                         "#squeue",
                         "42|1|csched-1-2|RUNNING|4|n01",
                         "#sqok", "#end", ""])
        with patch.object(csched, "sh", lambda *a, **k: (0, out, "")):
            v = csched.probe(host("cluster", backend="slurm", partition="cpu"), [])
        self.assertTrue(v["ok"])
        self.assertEqual((v["nodes"]["n01"]["gpus"], v["nodes"]["n01"]["gpus_used"]), (2, 0))
        self.assertEqual(v["squeue"][0]["job"], "42")
        self.assertTrue(v["squeue_ok"])
        self.assertEqual(csched.gpu_count("gpu:A100:2,gpu:1"), 3)

    def test_scripts_node_xargs_and_substitutions(self):
        spec = {"workdir": "/local/me/work", "cpus": 2, "mem_gb": 4, "ntasks": 5,
                "cmd": "python run.py --task {i} --arg {arg} --home {home} --host {host}"}
        with patch.object(csched, "LOCAL_HOME", "/local/me"):
            files = csched.scripts(spec, host(), "/jobs/1/2", 1, 2, 5, 3)
        run = files["run.sh"]
        self.assertIn("export CSCHED_JOB=1\n", run)
        self.assertIn("export CSCHED_CPUS=2\n", run)
        self.assertIn("export OMP_NUM_THREADS=2\n", run)
        self.assertIn("cd /home/me/work || exit 97", run)
        self.assertIn("--task ${CSCHED_TASK} --arg \"${CSCHED_ARG}\" --home /home/me --host node1", run)
        self.assertIn("xargs -P 3 -n 1 bash /jobs/1/2/task.sh", files["main.sh"])
        self.assertIn("sed -n \"$((K+1))p\" /jobs/1/2/tasks.txt", files["task.sh"])
        self.assertIn("echo $? > /jobs/1/2/rc/$i.tmp", files["task.sh"])

    def test_scripts_slurm_array_and_local_disk(self):
        spec = {"workdir": "/work", "cpus": 1, "mem_gb": 2, "ntasks": 4, "cmd": "true",
                "local_disk": "ssd0:10"}
        h = host("cluster", backend="slurm", local_disk="ssd0:20")
        files = csched.scripts(spec, h, "/jobs/1/2", 1, 2, 4, 2)
        self.assertIn("SLURM_ARRAY_TASK_ID", files["main.sh"])
        self.assertIn("K+=2", files["main.sh"])
        self.assertIn("export CSCHED_LOCAL_SSD0=/scratch/ssd0", files["run.sh"])


class RenderingTests(unittest.TestCase):
    def test_topo_str(self):
        self.assertEqual(csched.topo_str(None), "")
        self.assertEqual(csched.topo_str({"cores": 8, "threads": 16}), "8C16T")
        self.assertEqual(csched.topo_str({"cores": 24, "threads": 24, "p": 8, "e": 16}), "24C24T (8P+16E)")
        self.assertEqual(csched.topo_str({"cores": 8, "threads": 16,
                                          "nodes": {"a": {"cores": 4, "threads": 8},
                                                    "b": {"cores": 4, "threads": 8}}}), "8C16T: 2×4C8T")

    def test_host_lines_node_and_slurm(self):
        hosts = {"node1": host(), "cluster": host("cluster", backend="slurm", max_cpus=16)}
        views = {"node1": node_view(), "cluster": slurm_view()}
        lines = csched.host_lines(hosts, views, {"node1": {"cores": 8, "threads": 16}})
        self.assertIn("node1", lines[0])
        self.assertIn("free now 8 cpus", lines[0])
        self.assertIn("8C16T", lines[0])
        self.assertIn("ours   0/16 cpus (0 pending)", lines[1])
        self.assertIn("room 20", lines[2])
        self.assertEqual(csched.host_lines({"off": host("off", enabled=False)}, {})[0], "off       disabled")


class CLITests(unittest.TestCase):
    def test_init_submit_task_file_and_job_commands(self):
        with tempfile.TemporaryDirectory(prefix="csched-cli-") as tmp:
            root = Path(tmp)
            config = root / "config" / "csched.toml"
            env = {**os.environ, "CSCHED_CONFIG": str(config),
                   "CSCHED_STATE": str(root / "state"), "CSCHED_WORK": str(root / "work")}

            def cli(*args, success=True):
                r = subprocess.run([sys.executable, "-m", "gsched.csched", *args],
                                   cwd=root, env=env, text=True, capture_output=True, timeout=20)
                if success:
                    self.assertEqual(r.returncode, 0, f"{args}: {r.stdout}\n{r.stderr}")
                else:
                    self.assertNotEqual(r.returncode, 0, f"{args} unexpectedly succeeded")
                return r

            cli("init")
            self.assertTrue(config.is_file())
            self.assertIn("hosts.node1", config.read_text())
            (root / "tasks.txt").write_text("one\ntwo\nthree\n")
            self.assertEqual(cli("submit", "-n", "sweep", "-c", "2", "--task-file", "tasks.txt", "--",
                                 "python", "run.py", "{arg}").stdout.strip(), "1")
            job = json.loads(cli("show", "1").stdout.split("\nshard")[0])
            self.assertEqual((job["name"], job["ntasks"], job["state"]), ("sweep", 3, "queued"))
            self.assertEqual(job["spec"]["cpus"], 2)
            self.assertEqual(job["spec"]["mem_gb"], 4)
            self.assertEqual(job["spec"]["cmd"], "python run.py '{arg}'")
            self.assertTrue(Path(job["spec"]["args_file"]).is_file())
            self.assertIn("sweep", cli("ls").stdout)
            self.assertIn('"queued": 3', cli("show", "1").stdout)
            self.assertIn("cancelled", cli("cancel", "1").stdout)
            self.assertIn("3 tasks requeued", cli("retry", "1").stdout)
            cli("init", success=False)
            self.assertIn("--force", cli("init", success=False).stderr)
            cli("init", "--force")


if __name__ == "__main__":
    unittest.main()
