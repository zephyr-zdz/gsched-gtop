"""Small fixtures and isolated imports shared by the unittest modules."""
import atexit
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

# core captures these paths at import time. Keep them away from real user data,
# even when the suite is launched with GSCHED_* already set in the environment.
_import_dir = tempfile.TemporaryDirectory(prefix="gsched-tests-import-")
atexit.register(_import_dir.cleanup)
_root = Path(_import_dir.name)
with patch.dict(os.environ, {
    "GSCHED_CONFIG": str(_root / "hosts.toml"),
    "GSCHED_STATE": str(_root / "state"),
    "GSCHED_WORK": str(_root / "work"),
}):
    from gsched import core, gtop


def host(name="node", **overrides):
    return {"name": name, "backend": "node", "ssh": "local", "home": "/remote/home",
            "setup": "", "tier": 1, "optin": False, "enabled": True,
            "max_gpus": 4, "gpu_types": ["A100"], **overrides}


def card(index=0, **overrides):
    return {"index": index, "idle": True, "jobs": [], "foreign": [], "reserved": [],
            "total": 24 * 1024, "used": 0, **overrides}


def node_view(**overrides):
    return {"gpus": [card(0), card(1)], "mem_avail_gb": 64, "load1": 0,
            "ncpu": 16, **overrides}


def slurm_node(**overrides):
    return {"up": True, "free": {"A100": 2, "V100": 4}, "cpu_idle": 16,
            "mem_unalloc_gb": 64, **overrides}


def slurm_view(**overrides):
    return {"mine_gpus": 0, "pending": 0, "nodes": {"n01": slurm_node()}, **overrides}


def planned(gpus=None, **overrides):
    return {"gpus": [0] if gpus is None else gpus, "mem_gb": 0, "cpus": 1,
            "gpu_mem_gb": None, "client": "alice", **overrides}
