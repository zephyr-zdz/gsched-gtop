import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from support import gtop


class FormattingTests(unittest.TestCase):
    def test_expand_nodes(self):
        for text, expected in [
            (None, []), ("", []), ("node01", ["node01"]),
            ("gn[01-03,05],x1", ["gn01", "gn02", "gn03", "gn05", "x1"]),
            ("a[1-3],b[09-10]", ["a1", "a2", "a3", "b09", "b10"]),
        ]:
            with self.subTest(text=text):
                self.assertEqual(gtop.expand_nodes(text), expected)

    def test_bar_clamping_and_fractional_cell(self):
        for frac, width, expected in [
            (-1, 4, "····"), (0, 4, "····"), (0.5, 4, "██··"),
            (0.375, 4, "█▌··"), (1, 4, "████"), (2, 4, "████"), (0.5, 0, ""),
        ]:
            with self.subTest(frac=frac, width=width):
                self.assertEqual(gtop.bar(frac, width), expected)

    def test_spark_padding_recent_samples_and_saturation(self):
        self.assertEqual(gtop.spark([], 4), ("····", ""))
        self.assertEqual(gtop.spark([0, 50, 100], 5), ("··", " ▅█"))
        self.assertEqual(gtop.spark([1, 0, 25, 200], 3), ("", " ▃█"))
        self.assertEqual(gtop.spark(iter([12.5, 87.5]), 2), ("", "▂█"))

    def test_hms_boundaries(self):
        for seconds, expected in [(0, "0m00s"), (59.9, "0m59s"), (60, "1m00s"),
                                  (3599, "59m59s"), (3600, "1h00m"), (3660, "1h01m"),
                                  (86399, "23h59m"), (86400, "1d00h"), (176400, "2d01h")]:
            with self.subTest(seconds=seconds):
                self.assertEqual(gtop.hms(seconds), expected)

    def test_slurm_real_idx_numeric_order(self):
        self.assertEqual(gtop.slurm_real_idx({"step_gpus": "10,2", "gpuinfo": {"0": [], "1": []}}),
                         {"0": "2", "1": "10"})
        self.assertEqual(gtop.slurm_real_idx({"step_gpus": "3", "gpuinfo": {"0": []}}), {"0": "3"})

    def test_slurm_real_idx_missing_invalid_or_mismatched(self):
        for data in [{}, {"step_gpus": "GPU-uuid", "gpuinfo": {"0": []}},
                     {"step_gpus": "1,2", "gpuinfo": {"0": []}}]:
            with self.subTest(data=data):
                self.assertEqual(gtop.slurm_real_idx(data), {})


class CpuPageTests(unittest.TestCase):
    def poller(self, hosts, state, jobs, topo=None):
        class P:
            pass
        p = P()
        p.lock = threading.Lock()
        p.hosts, p.state, p.jobs = hosts, state, jobs
        p.t, p.err, p.topo = time.time(), None, topo or {}
        return p

    def render(self, lines):
        return "\n".join("".join(text for text, _ in segs) for segs in lines)

    def test_build_cpu_node_and_job_table(self):
        h = {"name": "node1", "backend": "node", "enabled": True, "pref": 1, "max_cpus": 8,
             "reserve_cpus": 0, "mem_reserve_gb": 0, "quiet_hours": "", "node_frac": 1.0,
             "cores_per_free_gpu": 0}
        view = {"ok": True, "ncpu": 8.0, "load": 2.0, "mem_avail_gb": 32.0, "ours_cpus": 2,
                "ours_mem_gb": 4, "nodes": {}, "squeue": [], "shards": {}}
        job = {"id": 3, "state": "running", "ntasks": 6, "counts": {"done": 2, "assigned": 2, "queued": 2},
               "cpus": 2, "hosts": ["node1"], "name": "sweep", "bad_hosts": "[]"}
        text = self.render(gtop.build_cpu(self.poller({"node1": h}, {"node1": view}, [job],
                                                      {"node1": {"cores": 8, "threads": 16}}), 120))
        self.assertIn("node1", text)
        self.assertIn("ours  2/8 cpus", text)
        self.assertIn("load █", text)
        self.assertIn("RAM avail    32G", text)
        self.assertIn("8C16T", text)
        self.assertIn("csched queue", text)
        self.assertIn("sweep", text)
        self.assertIn("done", text)
        self.assertIn("node1", text)

    def test_build_cpu_slurm_nodes_unreachable_and_config_error(self):
        h = {"name": "cluster", "backend": "slurm", "enabled": True, "pref": 2, "max_cpus": 16,
             "reserve_cpus": 0, "mem_reserve_gb": 0, "quiet_hours": "", "node_frac": 0.5,
             "cores_per_free_gpu": 0}
        node = {"up": True, "state": "mix", "cpu_idle": 12, "cpu_total": 16, "mem_total_gb": 64,
                "mem_unalloc_gb": 16, "gpus": 4, "gpus_used": 1}
        view = {"ok": True, "nodes": {"n01": node}, "squeue": [], "ours_cpus": 4, "ours_pending": 1,
                "ours_node": {"n01": 4}, "shards": {}, "ncpu": 0, "load": 0, "mem_avail_gb": 0}
        text = self.render(gtop.build_cpu(self.poller({"cluster": h}, {"cluster": view}, []), 120))
        self.assertIn("(1 pending)", text)
        self.assertIn("n01", text)
        self.assertIn("ours 4/8", text)
        self.assertIn("room 4", text)
        self.assertIn("GPU 3/4 free", text)
        down = dict(view, ok=False, err="ssh: timeout")
        text = self.render(gtop.build_cpu(self.poller({"cluster": h}, {"cluster": down}, []), 120))
        self.assertIn("UNREACHABLE: ssh: timeout", text)

    def test_cpu_poller_reports_missing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(gtop.C, "CONFIG", Path(tmp) / "missing.toml"):
                p = gtop.CpuPoller()
        self.assertEqual(p.hosts, {})
        p.once()
        self.assertIn("csched:", p.err)
