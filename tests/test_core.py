import contextlib
import io
import json
from pathlib import Path
import shlex
import sqlite3
import time
import unittest
from unittest.mock import patch

from support import core, host, card, node_view, slurm_node, slurm_view, planned


class ParsingTests(unittest.TestCase):
    def test_parse_gres(self):
        for text, expected in [
            ("(null)", {}), ("", {}), ("gpu:4", {"gpu": 4}),
            ("gpu:A100:2(S:0-1),gpu:A100:1(IDX:2),gpu:V100:4,shard:8",
             {"A100": 3, "V100": 4}),
        ]:
            with self.subTest(text=text):
                self.assertEqual(core.parse_gres(text), expected)

    def test_slurm_view_columns_and_queue(self):
        # Columns follow core.SINFO_FMT: node, GRES, used, state, CPU state,
        # memory, allocated memory, free memory, CPU load.
        p = {"nodes": "\n".join([
            "n01 gpu:A100:4 gpu:A100:1(IDX:0) mix 4/12/0/16 65536 16384 32768 2.5",
            "n02 gpu:V100:2 (null) idle 0/8/0/8 32768 0 N/A N/A",
            "n03 gpu:A100:2 (null) idle* 0/8/0/8 32768 0 16384 0",
            "n04 gpu:A100:2 (null) down 0/0/8/8 32768 0 16384 0",
            "short line",
        ]), "squeue": "42|RUNNING|n01|None\n43|PENDING|(null)|Resources\n"}
        v = core.slurm_view(p)
        self.assertEqual(v["free"], {"n01": {"A100": 3}, "n02": {"V100": 2}})
        r = v["nodes"]["n01"]
        self.assertEqual((r["cpu_alloc"], r["cpu_idle"], r["cpu_total"]), (4, 12, 16))
        self.assertEqual((r["mem_total_gb"], r["mem_unalloc_gb"], r["mem_free_gb"], r["load"]),
                         (64, 48, 32, 2.5))
        self.assertIsNone(v["nodes"]["n02"]["mem_free_gb"])
        self.assertIsNone(v["nodes"]["n02"]["load"])
        self.assertFalse(v["nodes"]["n03"]["up"])
        self.assertFalse(v["nodes"]["n04"]["up"])
        self.assertEqual(v["squeue"]["43"], ["43", "PENDING", "(null)", "Resources"])
        self.assertEqual(v["mine_gpus"], 0)

    def test_slurm_view_legacy_columns(self):
        v = core.slurm_view({"nodes": "n01 gpu:2 (null) idle", "squeue": ""})
        self.assertEqual(v["nodes"]["n01"]["cpu_idle"], 10**6)
        self.assertEqual(v["nodes"]["n01"]["mem_unalloc_gb"], 10**6)

    def test_quiet_hours_boundaries(self):
        for window, hour, minute, expected in [
            ("", 12, 0, False), ("09:00-17:00", 8, 59, False),
            ("09:00-17:00", 9, 0, True), ("09:00-17:00", 16, 59, True),
            ("09:00-17:00", 17, 0, False), ("22:00-06:00", 21, 59, False),
            ("22:00-06:00", 22, 0, True), ("22:00-06:00", 0, 30, True),
            ("22:00-06:00", 5, 59, True), ("22:00-06:00", 6, 0, False),
        ]:
            with self.subTest(window=window, hour=hour, minute=minute):
                now = time.mktime((2025, 1, 15, hour, minute, 0, 0, 0, -1))
                self.assertEqual(core.quiet_now({"quiet_hours": window}, now), expected)


class HostAndClientTests(unittest.TestCase):
    def setUp(self):
        self.hosts = {"main": host("main"), "opt": host("opt", optin=True),
                      "off": host("off", enabled=False),
                      "v100": host("v100", gpu_types=["V100"])}

    def test_allowed_default_and_fallback(self):
        self.assertEqual(core.allowed_hosts({}, self.hosts), ["main", "v100"])
        self.assertEqual(core.allowed_hosts({"fallback": True}, self.hosts), ["main", "opt", "v100"])

    def test_allowed_explicit_enabled_and_type(self):
        self.assertEqual(core.allowed_hosts({"hosts": ["opt", "off", "missing"]}, self.hosts), ["opt"])
        self.assertEqual(core.allowed_hosts({"gpu_type": "V100"}, self.hosts), ["v100"])
        self.assertEqual(core.allowed_hosts({"fallback": True, "gpu_type": "A100"}, self.hosts), ["main", "opt"])
        self.assertEqual(core.allowed_hosts({"hosts": ["main", "opt"], "fallback": True}, self.hosts), ["main", "opt"])

    def test_client_cap_counts_existing_and_planned_only_for_client(self):
        d = {"clients": {"alice": {"max_gpus": {"node": 4}}}}
        v = {"client_gpus": {"alice": 1, "bob": 99}}
        plan = {"node": [planned([0], client="alice"), planned([1], client="bob")]}
        self.assertTrue(core.within_client_cap({"client": "alice", "gpus": 2}, "node", v, d, plan))
        self.assertFalse(core.within_client_cap({"client": "alice", "gpus": 3}, "node", v, d, plan))
        self.assertTrue(core.within_client_cap({"client": "bob", "gpus": 100}, "node", v, d, plan))

    def test_client_cap_default_user_and_slurm_counts(self):
        with patch.object(core, "USER", "alice"):
            d = {"clients": {"alice": {"max_gpus": {"node": 2}}}}
            plan = {"node": [planned([], n=2)]}
            self.assertFalse(core.within_client_cap({}, "node", {}, d, plan))
            self.assertTrue(core.within_client_cap({}, "other", {}, d, plan))


class NodePlacementTests(unittest.TestCase):
    def test_idle_cards_and_planned_cards(self):
        self.assertEqual(core.place_node({"gpus": 2}, host(), node_view(), [], {})[:2], ("node", [0, 1]))
        self.assertEqual(core.place_node({}, host(), node_view(), [planned()], {})[:2], ("node", [1]))
        self.assertIsNone(core.place_node({"gpus": 2}, host(), node_view(), [planned()], {}))

    def test_max_gpus_includes_running_and_planned(self):
        for busy, taken in [(True, []), (False, [planned()])]:
            with self.subTest(busy=busy):
                cards = [card(0, idle=not busy, jobs=[{}] if busy else []), card(1)]
                self.assertIsNone(core.place_node({}, host(max_gpus=1), node_view(gpus=cards), taken, {}))
        self.assertIsNone(core.place_node({}, host(max_gpus=0), node_view(), [], {}))

    def test_host_ram_reserve_and_planned_ram(self):
        self.assertIsNotNone(core.place_node({"mem_gb": 48}, host(), node_view(), [], {}))
        self.assertIsNone(core.place_node({"mem_gb": 49}, host(), node_view(), [], {}))
        self.assertIsNone(core.place_node({"mem_gb": 40}, host(), node_view(), [planned(mem_gb=9)], {}))
        self.assertIsNotNone(core.place_node({"mem_gb": 60}, host(), node_view(), [], {"host_mem_reserve_gb": 4}))

    def test_cpu_load_and_planned_cpus(self):
        self.assertIsNotNone(core.place_node({"cpus": 4}, host(), node_view(load1=12), [], {}))
        self.assertIsNone(core.place_node({"cpus": 4}, host(), node_view(load1=13), [], {}))
        self.assertIsNone(core.place_node({"cpus": 4}, host(), node_view(load1=12), [planned(cpus=1)], {}))

    def test_pack_best_fit_before_idle_even_at_card_cap(self):
        jobs = [{"spec": json.dumps({"gpu_mem_gb": 8})}]
        cards = [card(0, idle=False, jobs=jobs, used=8*1024, total=32*1024),
                 card(1, idle=False, jobs=jobs, used=8*1024), card(2)]
        r = core.place_node({"gpu_mem_gb": 4}, host(max_gpus=2), node_view(gpus=cards), [], {})
        self.assertEqual(r[:2], ("node", [1]))

    def test_packing_limits_and_foreign_reserved_cards(self):
        jobs = [{"spec": json.dumps({"gpu_mem_gb": 8})}]
        for changes, defaults in [
            ({"foreign": ["other"]}, {}), ({"reserved": ["other"]}, {}),
            ({"jobs": [{"spec": "{}"}]}, {}), ({}, {"max_jobs_per_gpu": 1}),
            ({"used": 22*1024}, {}), ({}, {"pack_headroom_gb": 13}),
        ]:
            with self.subTest(changes=changes, defaults=defaults):
                g = card(0, idle=False, jobs=jobs, used=8*1024)
                g.update(changes)
                self.assertIsNone(core.place_node({"gpu_mem_gb": 4}, host(), node_view(gpus=[g]), [], defaults))

    def test_planned_memory_is_counted_for_packing(self):
        v = node_view(gpus=[card()])
        taken = [planned(gpu_mem_gb=12)]
        self.assertEqual(core.place_node({"gpu_mem_gb": 8}, host(), v, taken, {})[:2], ("node", [0]))
        self.assertIsNone(core.place_node({"gpu_mem_gb": 9}, host(), v, taken, {}))


class SlurmPlacementTests(unittest.TestCase):
    def setUp(self):
        self.h = host("cluster", backend="slurm", gpu_types=["V100", "A100"])

    def test_type_preference_and_explicit_type(self):
        self.assertEqual(core.place_slurm({}, self.h, slurm_view(), [])[1]["type"], "V100")
        self.assertEqual(core.place_slurm({"gpu_type": "A100"}, self.h, slurm_view(), [])[1]["type"], "A100")
        self.h["gres_types"] = {"V100": "tesla"}
        v = slurm_view(nodes={"n01": slurm_node(free={"tesla": 1})})
        self.assertEqual(core.place_slurm({}, self.h, v, [])[1]["type"], "V100")

    def test_resources_must_fit_on_one_node(self):
        v = slurm_view(nodes={"gpu": slurm_node(cpu_idle=1),
                              "cpu": slurm_node(free={}),
                              "ram": slurm_node(mem_unalloc_gb=1)})
        spec = {"gpus": 2, "cpus": 4, "mem_gb": 8}
        self.assertIsNone(core.place_slurm(spec, self.h, v, []))
        v["nodes"]["fits"] = slurm_node()
        self.assertEqual(core.place_slurm(spec, self.h, v, [])[1]["node"], "fits")

    def test_exclude_and_down_nodes(self):
        v = slurm_view(nodes={"n01": slurm_node(), "n02": slurm_node(up=False)})
        self.assertIsNone(core.place_slurm({"slurm_exclude": "n01,n03"}, self.h, v, []))

    def test_max_gpus_counts_existing_and_planned(self):
        self.assertIsNone(core.place_slurm({"gpus": 2}, self.h, slurm_view(mine_gpus=3), []))
        taken = [planned([], n=2, node="n01", type="A100")]
        self.assertIsNone(core.place_slurm({"gpus": 2}, self.h, slurm_view(mine_gpus=1), taken))

    def test_planned_resources_are_subtracted_per_node(self):
        for extra in [{"n": 2}, {"cpus": 16}, {"mem_gb": 64}]:
            with self.subTest(extra=extra):
                taken = {"node": "n01", "type": "A100", "n": 0, "cpus": 0, "mem_gb": 0, **extra}
                self.assertIsNone(core.place_slurm({"gpu_type": "A100", "mem_gb": 1}, self.h, slurm_view(), [taken]))

    def test_queue_ahead_requires_optin_and_respects_pending_cap(self):
        self.h["queue_ahead"] = 2
        v = slurm_view(nodes={})
        self.assertIsNone(core.place_slurm({}, self.h, v, []))
        r = core.place_slurm({"queue_ahead": True}, self.h, v, [])
        self.assertTrue(r[1]["ahead"])
        self.assertIsNone(r[1]["node"])
        self.assertEqual(r[1]["type"], "gpu")
        v["pending"] = 1
        self.assertIsNotNone(core.place_slurm({"queue_ahead": True}, self.h, v, []))
        self.assertIsNone(core.place_slurm({"queue_ahead": True}, self.h, v, [{"n": 1, "ahead": True}]))
        v["pending"] = 2
        self.assertIsNone(core.place_slurm({"queue_ahead": True}, self.h, v, []))
        self.h["max_gpus"] = 0
        self.assertIsNone(core.place_slurm({"queue_ahead": True}, self.h, slurm_view(nodes={}), []))


class PlacementTests(unittest.TestCase):
    def test_tier_fallback_wait_and_explicit_hosts(self):
        hosts = {"first": host("first"), "second": host("second", tier=2)}
        views = {"first": node_view(gpus=[]), "second": node_view()}
        d = {"fallback_after_min": 10}
        self.assertIsNone(core.place({}, 9.99, views, hosts, d, {}))
        self.assertEqual(core.place({}, 10, views, hosts, d, {}), ("second", [0]))
        self.assertEqual(core.place({"hosts": ["first", "second"]}, 0, views, hosts, d, {}), ("second", [0]))
        views["first"] = node_view()
        self.assertEqual(core.place({}, 100, views, hosts, d, {}), ("first", [0]))

    def test_balance_same_tier_by_card_load(self):
        hosts = {"busy": host("busy"), "empty": host("empty")}
        views = {"busy": node_view(gpus=[card(0, idle=False, jobs=[{}]), card(1)]), "empty": node_view()}
        self.assertEqual(core.place({}, 0, views, hosts, {}, {}), ("empty", [0]))
        self.assertEqual(core.place({}, 0, views, hosts, {}, {"empty": [planned()]}), ("busy", [1]))


class NodeViewTests(unittest.TestCase):
    def probe(self, **overrides):
        return {"gpus": "0, GPU-a, 100, 24576, 0", "procs": [], "user": "me", **overrides}

    def test_idle_debounce_and_busy_reset(self):
        p = self.probe()
        first = core.node_view({}, p, {}, {}, 1000, [])[0]
        self.assertFalse(first["idle"])
        self.assertEqual(first["idle_since"], 1000)
        self.assertFalse(core.node_view({}, p, {}, {"0": 1000}, 1089, [])[0]["idle"])
        self.assertTrue(core.node_view({}, p, {}, {"0": 1000}, 1090, [])[0]["idle"])
        p["gpus"] = "0, GPU-a, 100, 24576, 90"
        self.assertIsNone(core.node_view({}, p, {}, {"0": 1000}, 1100, [])[0]["idle_since"])

    def test_foreign_ignore_threshold(self):
        proc = {"uuid": "GPU-a", "user": "other", "pid": 1, "mib": 1000}
        p = self.probe(gpus="0, GPU-a, 1100, 24576, 0", procs=[proc])
        d = {"foreign_ignore_mib": 1024, "idle_mem_mib": 200, "idle_seconds": 0}
        g = core.node_view({}, p, d, {}, 1000, [])[0]
        self.assertTrue(g["idle"])
        self.assertEqual(g["foreign"], [])
        proc["mib"] = 1024
        g = core.node_view({}, p, d, {}, 1000, [])[0]
        self.assertFalse(g["idle"])
        self.assertEqual(g["foreign"], ["other"])

    def test_reserved_cards_and_owned_jobs(self):
        g = core.node_view({}, self.probe(), {"idle_seconds": 0}, {}, 1000, [], {0: ["bob", "bob", "alice"]})[0]
        self.assertFalse(g["idle"])
        self.assertEqual(g["reserved"], ["alice", "bob"])
        job = {"gpus": "[0]"}
        g = core.node_view({}, self.probe(), {"idle_seconds": 0}, {}, 1000, [job])[0]
        self.assertFalse(g["idle"])
        self.assertEqual(g["jobs"], [job])

    def test_nvidia_error_lines_skipped_and_unknown_util_busy(self):
        p = self.probe(gpus="Failed to initialize NVML: Driver/library version mismatch\n0, GPU-a, 100, 24576, N/A\nerror")
        gs = core.node_view({}, p, {"idle_seconds": 0}, {}, 1000, [])
        self.assertEqual(len(gs), 1)
        self.assertEqual(gs[0]["util"], 100)
        self.assertFalse(gs[0]["idle"])
        self.assertEqual(core.node_view({}, self.probe(gpus="NVIDIA-SMI has failed"), {}, {}, 1000, []), [])


class ReservationAndScriptTests(unittest.TestCase):
    def test_reservations_json_filters_own_user_and_unknown_host(self):
        payload = {"node": {"0": [core.USER, "other"], "1": [core.USER]}, "unknown": {"0": ["x"]}}
        d = {"reservations_cmd": "echo " + shlex.quote(json.dumps(payload))}
        self.assertEqual(core.reservations(d, {"node": {}, "empty": {}}),
                         {"node": {0: ["other"], 1: []}, "empty": {}})
        self.assertEqual(core.reservations({}, {"node": {}}), {"node": {}})

    def test_reservations_failure_and_invalid_json(self):
        for cmd in ["exit 3", "echo 'invalid JSON'"]:
            with self.subTest(cmd=cmd), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(core.reservations({"reservations_cmd": cmd}, {"node": {}}), {"node": {}})
                self.assertIn("reservations_cmd failed", output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as output:
                core.reservations({"reservations_cmd": cmd}, {"node": {}}, quiet=True)
                self.assertEqual(output.getvalue(), "")

    def test_scripts_exports_cuda_and_substitutions(self):
        with patch.object(core, "LOCAL_HOME", "/local/me"):
            h = host(setup="source {home}/env; echo {host}")
            spec = {"workdir": "/local/me/my work", "cmd": "echo {home} {workdir} {host}",
                    "env": {"MESSAGE": "a 'quoted' value", "COUNT": 3},
                    "setup_on": {"node": "echo {workdir}", "elsewhere": "wrong"}}
            run, wrap = core.scripts(spec, h, "/jobs/job dir", 7, [2, 5])
        self.assertIn("export GSCHED_JOB=7\n", run)
        self.assertIn("export GSCHED_HOST=node\n", run)
        self.assertIn("export GSCHED_JOBDIR='/jobs/job dir'\n", run)
        self.assertIn("export MESSAGE=" + shlex.quote(spec["env"]["MESSAGE"]) + "\n", run)
        self.assertIn("export COUNT=3\n", run)
        self.assertIn("export CUDA_DEVICE_ORDER=PCI_BUS_ID\n", run)
        self.assertIn("export CUDA_VISIBLE_DEVICES=2,5\n", run)
        self.assertIn("source /remote/home/env; echo node\necho /remote/home/my work", run)
        self.assertIn("cd '/remote/home/my work' || exit 97", run)
        self.assertIn("echo /remote/home /remote/home/my work node", run)
        self.assertNotIn("wrong", run)
        self.assertNotIn("{home}", run)
        self.assertIn("'/jobs/job dir'/run.sh", wrap)
        self.assertIn("rc.tmp", wrap)

    def test_slurm_scripts_preserve_slurm_cuda_environment(self):
        run, _ = core.scripts({"workdir": "/work", "cmd": "true"}, host(backend="slurm"), "/jobs/1", 1, [])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", run)
        self.assertNotIn("CUDA_DEVICE_ORDER", run)

    def test_remap_home_boundary_and_path_objects(self):
        with patch.object(core, "LOCAL_HOME", "/local/me"):
            for source, expected in [("/local/me", "/remote/home"),
                                     (Path("/local/me/project"), "/remote/home/project"),
                                     ("/local/me2/project", "/local/me2/project"),
                                     ("/data/file", "/data/file")]:
                with self.subTest(source=source):
                    self.assertEqual(core.remap(source, host()), expected)


class LocalDiskAndTimeTests(unittest.TestCase):
    def test_parse_localdisk(self):
        self.assertEqual(core.parse_localdisk("localdisk:ssd0:400G"), {"ssd0": 400})
        self.assertEqual(core.parse_localdisk("gpu:A100:2,localdisk:ssd0:50G,localdisk:hdd0:1000G"),
                         {"ssd0": 50, "hdd0": 1000})
        self.assertEqual(core.parse_localdisk("(null)"), {})

    def test_local_disk_spec_has_no_default_pool(self):
        slurm = host("cluster", backend="slurm")
        self.assertEqual(core.local_disk_spec({}, slurm), [])
        self.assertEqual(core.local_disk_spec({}, host("box")), [])
        self.assertEqual(core.local_disk_spec({"local_disk": "ssd0:50"}, host("box")), [])
        self.assertEqual(core.local_disk_spec({"local_disk": "ssd0:50"}, slurm), [("ssd0", 50)])
        self.assertEqual(core.local_disk_spec({"local_disk": "ssd0:10,hdd0:20"}, slurm),
                         [("ssd0", 10), ("hdd0", 20)])
        self.assertEqual(core.local_disk_spec({"local_disk": "none"}, host("cluster", backend="slurm",
                                                                          local_disk="ssd0:50")), [])
        self.assertEqual(core.local_disk_spec({}, host("cluster", backend="slurm", local_disk="ssd0:50")),
                         [("ssd0", 50)])

    def test_slurm_view_parses_localdisk_pools(self):
        v = core.slurm_view({"nodes": "n01 gpu:A100:2,localdisk:ssd0:400G localdisk:ssd0:100G idle 0/8/0/8 "
                                     "32768 0 N/A N/A", "squeue": ""})
        self.assertEqual(v["nodes"]["n01"]["disk_total"], {"ssd0": 400})
        self.assertEqual(v["nodes"]["n01"]["disk_free"], {"ssd0": 300})

    def test_place_slurm_requires_disk_room_and_records_it(self):
        h = host("cluster", backend="slurm", gpu_types=["A100"], local_disk="ssd0:50")
        v = slurm_view(nodes={"n01": slurm_node(disk_total={"ssd0": 400}, disk_free={"ssd0": 40})})
        self.assertIsNone(core.place_slurm({}, h, v, []))
        v["nodes"]["n01"]["disk_free"]["ssd0"] = 100
        r = core.place_slurm({}, h, v, [])
        self.assertEqual(r[1]["disk"], {"ssd0": 50})
        taken = [{"node": "n01", "type": "A100", "n": 0, "cpus": 0, "mem_gb": 0, "disk": {"ssd0": 60}}]
        self.assertIsNone(core.place_slurm({}, h, v, taken))

    def test_time_secs_and_stem(self):
        for text, expected in [("30", 1800), ("1:30", 90), ("2:03:04", 7384),
                               ("1-00:00:00", 86400), ("2-03", 183600), ("0-00:30", 30)]:
            with self.subTest(text=text):
                self.assertEqual(core.time_secs(text), expected)
        for name, stem in [("run", "run"), ("run-4", "run"), ("run-12", "run"), ("a_03b", "a"),
                           ("tsal-l2-vrand0-irc-12", "tsal-l2-vrand0-irc")]:
            with self.subTest(name=name):
                self.assertEqual(core.stem_of(name), stem)

    def test_runtime_estimate_from_same_stem(self):
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        c.execute("create table jobs(state text, rc integer, host text, started real, ended real, name text)")
        h = host("cluster", backend="slurm", max_time="1-00:00:00")
        self.assertIsNone(core.est_secs(c, "run-1", h))
        c.executemany("insert into jobs values('done',0,'cluster',?,?,?)",
                      [(0, 1000, "run-3"), (0, 2000, "run-8"), (0, 99999, "other-1")])
        self.assertEqual(core.est_secs(c, "run-12", h), 3600)  # 1.5 × 2000 s, rounded up to 1 h
        self.assertEqual(core.est_time(c, "run-12", h), "1:00:00")
        self.assertIsNone(core.est_secs(c, "run-12", host("box")))

    def test_slurm_argv_name_comment_disk_and_time_clamp(self):
        c = sqlite3.connect(":memory:")
        c.execute("create table jobs(state text, rc integer, host text, started real, ended real, name text)")
        h = host("cluster", backend="slurm", partition="gpu", gpu_types=["A100"], max_time="1-00:00:00",
                 local_disk="ssd0:50", default_time="8:00:00")
        j = {"id": 7, "name": "my job/1", "spec": json.dumps({"cpus": 2, "mem_gb": 3, "time": "2-00:00:00"}),
             "client": "alice"}
        args = core.slurm_argv(c, j, h, "/jobs/7", {"type": "A100", "n": 2})
        self.assertEqual(args[args.index("-J") + 1], "myjob1")
        self.assertEqual(args[args.index("--comment") + 1], "gsched #7 alice")
        self.assertIn("--gres=gpu:A100:2,localdisk:ssd0:50G", args)
        self.assertEqual(args[args.index("-t") + 1], "1-00:00:00")  # clamped by max_time
        j["spec"] = json.dumps({"cpus": 2})
        self.assertEqual(core.slurm_argv(c, j, h, "/jobs/7", {"type": "A100", "n": 1})[
            core.slurm_argv(c, j, h, "/jobs/7", {"type": "A100", "n": 1}).index("-t") + 1], "8:00:00")

    def test_scripts_export_local_disk_only_when_configured(self):
        spec = {"workdir": "/work", "cmd": "true", "local_disk": "ssd0:50", "env": {"TMPDIR": "/mine"}}
        run, _ = core.scripts(spec, host("cluster", backend="slurm"), "/jobs/1", 1, [])
        self.assertIn("export GSCHED_LOCAL_SSD0=/scratch/ssd0", run)
        self.assertIn("export GSCHED_LOCAL=/scratch/ssd0", run)
        self.assertIn("export PYSCF_TMPDIR=/scratch/ssd0/tmp", run)
        self.assertNotIn("export TMPDIR=/scratch/ssd0/tmp", run)  # the job set it itself
        run, _ = core.scripts({"workdir": "/work", "cmd": "true"}, host("box"), "/jobs/1", 1, [])
        self.assertNotIn("GSCHED_LOCAL", run)
