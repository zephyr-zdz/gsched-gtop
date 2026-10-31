"""gsched: a small GPU job queue for a handful of servers you reach over ssh (+ Slurm clusters as overflow).

The controller runs on one machine (stdlib only, Python >= 3.11). It polls every host over ssh, places queued jobs
on idle GPUs (or packs jobs that declare --gpu-mem onto GPUs gsched already uses), balancing across hosts of the
same tier, and tracks them to completion. Other users' processes are only observed, never touched.
Jobs carry a client (submit --client, default the OS user); [clients.<name>] max_gpus = {host = n} caps a client's
cards per host inside the host's own max_gpus.

  gsched init                        write an example config to ~/.config/gsched/hosts.toml
  gsched submit [opts] -- CMD...     queue a job (prints its id)
  gsched ls [-a] [--client NAME]     jobs (active; -a: all)
  gsched status [--client NAME]      GPUs per host: idle / mine / foreign
  gsched plan                        what the next tick would start where (no side effects)
  gsched show ID | logs ID [-f] | cancel ID... | retry ID...
  gsched tick                        one scheduling pass (the daemon does this every poll_seconds)
  gsched daemon                      loop forever (e.g. as a systemd user unit)
"""
import argparse, concurrent.futures as cf, fcntl, getpass, json, os, re, shlex, sqlite3, subprocess, sys, time, tomllib
from pathlib import Path

from . import __version__

_xdg = lambda var, default: Path(os.environ.get(var) or Path.home() / default)
CONFIG = Path(os.environ.get("GSCHED_CONFIG") or _xdg("XDG_CONFIG_HOME", ".config") / "gsched/hosts.toml")
STATE = Path(os.environ.get("GSCHED_STATE") or _xdg("XDG_STATE_HOME", ".local/state") / "gsched")
WORK = Path(os.environ.get("GSCHED_WORK") or STATE / "jobs")  # local copies of job logs (and the jobs of ssh = "local")
LOCAL_HOME = str(Path.home())
USER = getpass.getuser()
ACTIVE = ("queued", "staging", "running", "fetching")


# ---------------------------------------------------------------- config / db

def load_config():
    if not CONFIG.exists():
        sys.exit(f"no config at {CONFIG}: run `gsched init`, or point GSCHED_CONFIG at your hosts.toml")
    cfg = tomllib.loads(CONFIG.read_text())
    d = cfg.get("defaults", {})
    hosts = {}
    for name, h in cfg["hosts"].items():
        h = {**h, "name": name}
        h.setdefault("backend", "node")
        h.setdefault("ssh", name)
        h.setdefault("ssh_opts", [])
        h.setdefault("tier", 1)
        h.setdefault("optin", False)
        h.setdefault("enabled", True)        # false: never place jobs here (still shown in status)
        h.setdefault("quiet_hours", "")      # "01:00-08:00": no new jobs then, running ones are SIGSTOPped (resumed after)
        h.setdefault("max_gpus", 1 << 30)
        h.setdefault("home", LOCAL_HOME)
        h.setdefault("jobroot", remap(WORK, h))  # job dirs: the local WORK path, under the host's home
        h.setdefault("setup", "")
        h.setdefault("bwlimit", d.get("bwlimit", "0"))
        hosts[name] = h
    # [clients.<name>] max_gpus = {host = n}: per-client card caps (a client without an entry for a host: only max_gpus)
    d["clients"] = {k: {"max_gpus": dict(v.get("max_gpus", {}))} for k, v in cfg.get("clients", {}).items()}
    return d, hosts


def db():
    STATE.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(STATE / "gsched.db", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("pragma journal_mode=wal")
    c.execute("""create table if not exists jobs(
        id integer primary key, name text, spec text, state text, host text, gpus text, handle text,
        jobdir text, submitted real, started real, ended real, rc integer, attempts integer default 0, note text)""")
    c.execute("create table if not exists hoststate(host text primary key, data text, ts real)")
    if "client" not in [r[1] for r in c.execute("pragma table_info(jobs)")]:
        c.execute("alter table jobs add column client text")  # who submitted it (submit --client), for per-client caps
        c.execute("update jobs set client=? where client is null", (USER,))
        c.commit()
    return c


def client_of(j):
    """A job row's client (the OS user when it has none)."""
    return (j["client"] if "client" in j.keys() else None) or USER


def upd(c, jid, **kw):
    c.execute(f"update jobs set {', '.join(k + '=?' for k in kw)} where id=?", (*kw.values(), jid))
    c.commit()


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


# ---------------------------------------------------------------- remote execution

def ssh_argv(h):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={Path.home()}/.ssh/gsched-%C", "-o", "ControlPersist=5m",
            *h["ssh_opts"]]


def sh(h, cmd, stdin=None, timeout=120):
    argv = ["bash", "-c", cmd] if h["ssh"] == "local" else [*ssh_argv(h), h["ssh"], cmd]
    try:
        r = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    return r.returncode, r.stdout, r.stderr


def homeform(path):
    """Absolute path, with targets of symlinks in ~ (e.g. ~/data -> /mnt/big/me) folded back under ~, so they remap
    to the same ~/... path on every host."""
    path = os.path.abspath(path)
    try:
        entries = list(Path(LOCAL_HOME).iterdir())
    except OSError:
        return path
    for e in entries:
        if e.is_symlink():
            real = os.path.realpath(e)
            if path == real or path.startswith(real + "/"):
                return str(e) + path[len(real):]
    return path


def remap(path, h):
    """Local path -> the same path under the host's home (e.g. /home/me/x -> /home/me2/x where home = "/home/me2")."""
    path = str(path)
    if path == LOCAL_HOME or path.startswith(LOCAL_HOME + "/"):
        return h["home"] + path[len(LOCAL_HOME):]
    return path


def rsync_target(h, path):
    return path if h["ssh"] == "local" else f"{h['ssh']}:{path}"


PROBE = r'''
import json, os, pwd, subprocess, sys
a = json.loads(sys.argv[1])
def q(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""
out = {"jobs": {}}
for jid, d in a["jobs"].items():
    j = {"rc": None, "alive": False}
    try:
        j["rc"] = int(open(d + "/rc").read().strip())
    except Exception:
        pass
    try:
        j["start"] = float(open(d + "/start").read().strip())  # written by wrap.sh when the job really starts
    except Exception:
        pass
    try:
        pid = open(d + "/pid").read().strip()
        j["alive"] = (d + "/wrap.sh") in open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except Exception:
        pass
    out["jobs"][jid] = j
if a["backend"] == "slurm":
    out["nodes"] = q("sinfo", "-N", "-h", "-p", a["partition"], "-O", a["sinfo_fmt"])
    try:  # a failing squeue (controller hiccup) must not look like "all our jobs vanished"
        r = subprocess.run(["squeue", "-u", pwd.getpwuid(os.getuid()).pw_name, "-h", "-o", "%i|%T|%N|%r"],
                           capture_output=True, text=True, timeout=60)
        out["squeue"], out["squeue_ok"] = r.stdout, r.returncode == 0
    except Exception:
        out["squeue"], out["squeue_ok"] = "", False
else:
    out["gpus"] = q("nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits")
    procs = []
    for line in q("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                  "--format=csv,noheader,nounits").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) != 3 or not p[1].isdigit():
            continue
        try:
            user = pwd.getpwuid(os.stat(f"/proc/{p[1]}").st_uid).pw_name
        except Exception:
            user = "?"
        procs.append({"uuid": p[0], "pid": int(p[1]), "mib": int(p[2]) if p[2].isdigit() else 0, "user": user})
    out["procs"] = procs
    mi = dict(l.split(":", 1) for l in open("/proc/meminfo"))
    out["mem_avail_gb"] = int(mi["MemAvailable"].split()[0]) / 2**20
    out["load1"] = os.getloadavg()[0]
    out["ncpu"] = os.cpu_count()
    out["user"] = pwd.getpwuid(os.getuid()).pw_name
print(json.dumps(out))
'''


def probe(h, jobdirs):
    arg = json.dumps({"backend": h["backend"], "partition": h.get("partition", ""), "sinfo_fmt": SINFO_FMT,
                      "jobs": jobdirs})
    rc, out, err = sh(h, f"python3 - {shlex.quote(arg)}", stdin=PROBE, timeout=90)
    if rc:
        return {"error": (err or out).strip()[-300:] or f"rc={rc}"}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        return {"error": f"bad probe output: {e}: {out[-200:]}"}


# per-node columns read by slurm_view (also used by gtop's Slurm monitor)
SINFO_FMT = "NodeList:60,Gres:400,GresUsed:400,StateCompact:20,CPUsState:30,Memory:20,AllocMem:20,FreeMem:20,CPUsLoad:20"
GRES = re.compile(r"gpu:(?:([^:,()]+):)?(\d+)")


def parse_gres(s):
    r = {}
    for t, n in GRES.findall(s):
        r[t or "gpu"] = r.get(t or "gpu", 0) + int(n)
    return r


LOCALDISK = re.compile(r"localdisk:([A-Za-z0-9_]+):(\d+)([KMGTP]?)")


def parse_localdisk(s):
    """Node-local scratch pools in sinfo Gres/GresUsed ("localdisk:ssd0:400G") -> {pool: GiB}."""
    return {pool: int(n) * 1024 ** "KMGTP".find(u) / 1024 ** 2 if u else int(n) / 1024 ** 3
            for pool, n, u in LOCALDISK.findall(s)}


def local_disk_spec(spec, h):
    """The job's node-local scratch request on a Slurm host: [(pool, GiB), ...]. --local-disk beats the host default
    `local_disk`; "none" turns it off, and there is no default pool unless the host configures one.
    Format "ssd0:50" or "ssd0:50,hdd0:500"."""
    v = spec.get("local_disk") or h.get("local_disk") or "none"
    if h["backend"] != "slurm" or v == "none":
        return []
    return [(pool, int(gb)) for pool, gb in (x.split(":") for x in v.split(","))]


# ---------------------------------------------------------------- capacity model

def node_view(h, p, d, prev, now, mine_jobs, reserved=None):
    """Per-GPU view of a node host: idle (debounced), packable, my gsched jobs on it.
    `reserved`: {gpu index: [holders other than us]} from reservations_cmd; a reserved card is never idle."""
    reserved = reserved or {}
    gpus = []
    for line in p["gpus"].strip().splitlines():
        if line.count(",") != 4:  # nvidia-smi error text (e.g. driver/library mismatch): host shows no usable GPUs
            continue
        i, uuid, used, total, util = (x.strip() for x in line.split(","))
        procs = [x for x in p["procs"] if x["uuid"] == uuid]
        foreign = [x for x in procs if x["user"] != p["user"]]
        ignored = sum(x["mib"] for x in foreign if x["mib"] < d.get("foreign_ignore_mib", 1024))
        blocking = [x for x in foreign if x["mib"] >= d.get("foreign_ignore_mib", 1024)]
        big = [x for x in procs if x["mib"] >= d.get("foreign_ignore_mib", 1024)]
        used, total = int(used), int(total)
        util = int(util) if util.isdigit() else 100
        jobs = [j for j in mine_jobs if int(i) in json.loads(j["gpus"] or "[]")]
        held = reserved.get(int(i), [])
        idle_now = (not jobs and not big and not held
                    and used - ignored < d.get("idle_mem_mib", 2000) and util < d.get("idle_util", 10))
        since = prev.get(i) if idle_now else None
        since = since if since is not None else (now if idle_now else None)
        gpus.append({"index": int(i), "used": used, "total": total, "util": util, "jobs": jobs,
                     "foreign": sorted({x["user"] for x in blocking}),
                     "reserved": sorted(set(held)),
                     "mine_other": [x for x in procs if x["user"] == p["user"]] if not jobs else [],
                     "idle_since": since,
                     "idle": idle_now and now - since >= d.get("idle_seconds", 90)})
    return gpus


def quiet_now(h, now=None):
    """True inside the host's quiet_hours ("HH:MM-HH:MM", the controller's local time; may wrap past midnight)."""
    if not h.get("quiet_hours"):
        return False
    a, b = (int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t in h["quiet_hours"].split("-"))
    t = time.localtime(now)
    m = t.tm_hour * 60 + t.tm_min
    return a <= m < b if a <= b else m >= a or m < b


def quiet_signal(c, hosts):
    """Quiet hours (e.g. a noisy workstation at night): SIGSTOP the process group of every running
    job on a quiet node host (resent each pass, so late starters stop too) and SIGCONT it once the window is over.
    The job keeps its GPU memory but puts no load on the card; "paused" in the note marks jobs to continue."""
    for j in c.execute("select * from jobs where state='running' order by id").fetchall():
        h = hosts.get(j["host"])
        if not h or h["backend"] != "node" or not j["jobdir"]:
            continue
        quiet, paused = quiet_now(h), (j["note"] or "").startswith("paused")
        if not quiet and not paused:
            continue
        q = shlex.quote(j["jobdir"])
        rc, _, err = sh(h, f"p=$(cat {q}/pid) && grep -q wrap.sh /proc/$p/cmdline && kill -{'STOP' if quiet else 'CONT'} -- -$p")
        if quiet and not paused and not rc:
            upd(c, j["id"], note=f"paused: quiet hours {h['quiet_hours']}")
            log(f"job {j['id']} paused on {h['name']} (quiet hours {h['quiet_hours']})")
        elif not quiet and not rc:
            upd(c, j["id"], note="")
            log(f"job {j['id']} resumed on {h['name']} (quiet hours over)")
        elif rc and not quiet:
            log(f"job {j['id']}: SIGCONT on {h['name']} failed: {err.strip()[-200:]}")


def gsched_gpus(gpus):
    return sum(1 for g in gpus if g["jobs"])


def needs_sync(spec, h):
    return h["ssh"] != "local" and bool(spec.get("sync", True) or spec.get("sync_paths"))


def allowed_hosts(spec, hosts):
    """Hosts a job may ever run on: --hosts or the non-opt-in ones (+ opt-in with --fallback), enabled, right card type."""
    allowed = spec.get("hosts") or [n for n, h in hosts.items() if not h["optin"] or spec.get("fallback")]
    if spec.get("fallback"):
        allowed = list(dict.fromkeys(allowed + [n for n, h in hosts.items() if h["optin"]]))
    allowed = [n for n in allowed if n in hosts and hosts[n]["enabled"]]
    if spec.get("gpu_type"):  # a card-type requirement keeps the job on hosts that list that type (nodes too)
        allowed = [n for n in allowed if spec["gpu_type"] in hosts[n].get("gpu_types", [])]
    return allowed


def place(spec, waited_min, views, hosts, d, plan, syncing=()):
    """Return (host, gpus|gres) for a job, or None. `plan` holds placements made earlier in this pass;
    hosts in `syncing` already have an rsync running, so jobs that must sync wait for the next pass."""
    allowed = [n for n in allowed_hosts(spec, hosts) if n in views and not (n in syncing and needs_sync(spec, hosts[n]))
               and not quiet_now(hosts[n])]
    tiers = sorted({hosts[n]["tier"] for n in allowed})
    explicit = bool(spec.get("hosts"))
    for k, tier in enumerate(tiers):
        if k > 0 and not explicit and waited_min < d.get("fallback_after_min", 10):
            break
        cands = []
        for n in allowed:
            h, v = hosts[n], views[n]
            if h["tier"] != tier:
                continue
            taken = plan.get(n, [])
            if h["backend"] == "slurm":
                r = place_slurm(spec, h, v, taken)
            else:
                r = place_node(spec, h, v, taken, d)
            if r and within_client_cap(spec, n, v, d, plan, r[1]):
                cands.append(r)
        if cands:
            cands.sort(key=lambda r: r[2])
            return cands[0][0], cands[0][1]
    return None


def within_client_cap(spec, n, v, d, plan, gpus=None):
    """The job's client stays within its [clients.<name>] max_gpus on host n (counting this pass' placements).
    `gpus`: the node cards it would get; then distinct cards count, so packing onto a card the client already
    holds adds nothing. Otherwise (Slurm) the job's GPU count is added."""
    client = spec.get("client") or USER
    cap = d.get("clients", {}).get(client, {}).get("max_gpus", {}).get(n)
    if cap is None:
        return True
    taken = [t for t in plan.get(n, []) if t.get("client") == client]
    if isinstance(gpus, list):
        cards = set(v.get("client_cards", {}).get(client, ())) | {i for t in taken for i in t["gpus"]} | set(gpus)
        return len(cards) <= cap
    used = v.get("client_gpus", {}).get(client, 0) + sum(t.get("n", len(t["gpus"])) for t in taken)
    return used + spec.get("gpus", 1) <= cap


def place_node(spec, h, v, taken, d):
    n = h["name"]
    if v["mem_avail_gb"] - spec.get("mem_gb", 0) - sum(t["mem_gb"] for t in taken) < d.get("host_mem_reserve_gb", 16):
        return None
    if v["load1"] + spec.get("cpus", 1) + sum(t["cpus"] for t in taken) > v["ncpu"]:
        return None
    gpus = v["gpus"]
    planned = {g["index"]: [t for t in taken if g["index"] in t["gpus"]] for g in gpus}
    in_use = sum(1 for g in gpus if g["jobs"] or planned[g["index"]])
    load = in_use / max(h["max_gpus"], 1)
    idle = [g for g in gpus if g["idle"] and not planned[g["index"]]]
    if spec.get("gpu_mem_gb"):  # a declared memory need also filters whole idle cards (hosts with mixed cards)
        idle = [g for g in idle if g["total"] - g["used"] >= spec["gpu_mem_gb"] * 1024]
    want = spec.get("gpus", 1)
    if spec.get("gpu_mem_gb") and want == 1:
        need = spec["gpu_mem_gb"] * 1024
        best = None
        for g in gpus:  # best fit on GPUs that already hold only packable (--gpu-mem) gsched jobs
            mems = [json.loads(r["spec"]).get("gpu_mem_gb") for r in g["jobs"]] + [t["gpu_mem_gb"] for t in planned[g["index"]]]
            if not mems or not all(mems) or g["foreign"] or g["reserved"] or len(mems) >= d.get("max_jobs_per_gpu", 4):
                continue
            planned_mib = sum(t["gpu_mem_gb"] for t in planned[g["index"]]) * 1024
            free = min(g["total"] - g["used"] - planned_mib, g["total"] - sum(mems) * 1024)
            left = free - need - d.get("pack_headroom_gb", 4) * 1024
            if left >= 0 and (best is None or left < best[1]):
                best = (g, left)
        if best:
            return n, [best[0]["index"]], (0, best[1], load)
    if len(idle) >= want and in_use + want <= h["max_gpus"]:
        return n, [g["index"] for g in idle[:want]], (1, load, -len(idle))
    return None


def place_slurm(spec, h, v, taken):
    # at most max_pending (default: queue_ahead, else 2) of our Slurm jobs may sit PENDING on this host, whatever
    # the reason: even a free-card placement can end up PENDING, because Slurm gives the card to a higher-priority
    # pending job. Placements planned earlier in this pass count too (they may pend), so at most this many new
    # sbatches go out per pass — fine, the daemon ticks every poll_seconds. Queue properly, don't squat on cards.
    if v["pending"] + len(taken) >= (h.get("max_pending") or h.get("queue_ahead") or 2):
        return None
    mine = v["mine_gpus"] + sum(t["n"] for t in taken)
    want = spec.get("gpus", 1)
    if mine + want > h["max_gpus"]:
        return None
    types = [spec["gpu_type"]] if spec.get("gpu_type") else h.get("gpu_types") or ["gpu"]
    cpus, mem = spec.get("cpus", 4), spec.get("mem_gb", 0)
    skip = set((spec.get("slurm_exclude") or "").split(","))
    disk = local_disk_spec(spec, h)
    for rank, t in enumerate(types):
        for node, r in sorted(v["nodes"].items()):  # one node must hold the GPUs, CPUs and RAM together
            if node in skip or not r["up"]:
                continue
            here = [x for x in taken if x.get("node") == node]
            free = r["free"].get(h.get("gres_types", {}).get(t, t), 0) - sum(x["n"] for x in here if x["type"] == t)
            if (free >= want and r["cpu_idle"] - sum(x.get("cpus", 0) for x in here) >= cpus
                    and r["mem_unalloc_gb"] - sum(x.get("mem_gb", 0) for x in here) >= mem
                    and all(r.get("disk_free", {}).get(pool, 0) - sum(x.get("disk", {}).get(pool, 0) for x in here) >= gb
                            for pool, gb in disk)):
                return h["name"], {"type": t, "n": want, "node": node, "cpus": cpus, "mem_gb": mem,
                                   "disk": dict(disk)}, (2, rank, 0)
    # queue ahead (opt-in per job, capped per host by `queue_ahead`): no card free now, so get in line in Slurm instead
    # of racing other users' pending jobs for the next freed card; any allowed type, Slurm picks the node
    if spec.get("queue_ahead") and h.get("queue_ahead", 0):
        pend = v.get("pending", 0) + sum(1 for x in taken if x.get("ahead"))
        if pend < h["queue_ahead"]:
            t = spec.get("gpu_type") or "gpu"
            return h["name"], {"type": t, "n": want, "node": None, "cpus": cpus, "mem_gb": mem, "ahead": True}, (3, 0, 0)
    return None


def slurm_view(p):
    """Per node: free GPUs by type, idle CPUs, RAM (MiB from sinfo -> GiB) and whether Slurm may place there."""
    nodes = {}
    for line in p["nodes"].strip().splitlines():
        f = line.split()
        if len(f) < 4:
            continue
        tot, used = parse_gres(f[1]), parse_gres(f[2])
        r = {"state": f[3], "up": any(s in f[3] for s in ("idle", "mix")) and "*" not in f[3],  # *: not responding
             "free": {t: tot[t] - used.get(t, 0) for t in tot}, "gres": tot}
        dt, du = parse_localdisk(f[1]), parse_localdisk(f[2])
        r["disk_total"], r["disk_free"] = dt, {k: dt[k] - du.get(k, 0) for k in dt}
        try:
            a, i, o, t = (int(x) for x in f[4].split("/"))
            r.update(cpu_alloc=a, cpu_idle=i, cpu_total=t,
                     mem_total_gb=int(f[5]) / 1024, mem_unalloc_gb=(int(f[5]) - int(f[6])) / 1024)
            r["mem_free_gb"] = int(f[7]) / 1024 if f[7].isdigit() else None
            r["load"] = float(f[8]) if len(f) > 8 and f[8].replace(".", "", 1).isdigit() else None
        except (IndexError, ValueError):
            r.update(cpu_alloc=0, cpu_idle=10**6, cpu_total=0, mem_total_gb=0, mem_unalloc_gb=10**6, mem_free_gb=None,
                     load=None)  # old probe without these columns: don't block placement on them
        nodes[f[0]] = r
    free = {n: r["free"] for n, r in nodes.items() if r["up"]}
    return {"free": free, "nodes": nodes, "mine_gpus": 0, "squeue_ok": p.get("squeue_ok", True),
            "squeue": {l.split("|")[0]: l.split("|") for l in p["squeue"].strip().splitlines()}}


# ---------------------------------------------------------------- job lifecycle

def scripts(spec, h, jobdir, jid, gpus):
    wd = remap(spec["workdir"], h)
    env = {"GSCHED_JOB": str(jid), "GSCHED_HOST": h["name"], "GSCHED_JOBDIR": jobdir, **spec.get("env", {})}
    if h["backend"] == "node":
        env |= {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))}
    cmd = spec["cmd"].replace("{home}", h["home"]).replace("{workdir}", wd).replace("{host}", h["name"])
    setup = "\n".join(s for s in (h["setup"], spec.get("setup_on", {}).get(h["name"], "")) if s)
    setup = setup.replace("{home}", h["home"]).replace("{workdir}", wd).replace("{host}", h["name"])  # as in the command
    local = ""
    for i, (pool, _) in enumerate(local_disk_spec(spec, h)):
        # each requested pool is mounted at /scratch/<pool>, private to the job and deleted when it ends.
        # GSCHED_LOCAL = the first pool; TMPDIR and QM scratch go there unless the job set them itself.
        d = f"/scratch/{pool}"
        local += f"export GSCHED_LOCAL_{pool.upper()}={d}\n"
        if i == 0:
            tmp = {k: f"{d}/tmp" for k in ("TMPDIR", "PYSCF_TMPDIR", "GAUSS_SCRDIR", "QCSCRATCH", "PSI_SCRATCH")
                   if k not in spec.get("env", {})}
            local += (f"if mkdir -p {d}/tmp 2>/dev/null; then export GSCHED_LOCAL={d}\n"
                      + "".join(f"  export {k}={v}\n" for k, v in tmp.items())
                      + f"else echo 'gsched: {d} not writable, using defaults' >&2; fi\n")
    run = ("#!/bin/bash\n" + "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in env.items()) + local
           + f"{setup}\ncd {shlex.quote(wd)} || exit 97\n{cmd}\n")
    q = shlex.quote(jobdir)
    wrap = (f"#!/bin/bash\necho $$ > {q}/pid\ndate +%s > {q}/start\ntrap true TERM INT HUP\n"
            f"bash -l {q}/run.sh > {q}/out.log 2>&1\necho $? > {q}/rc.tmp && mv {q}/rc.tmp {q}/rc\n")
    return run, wrap


def bg(argv, rcfile, logfile):
    """Run argv locally in the background; its exit code lands in rcfile."""
    Path(rcfile).unlink(missing_ok=True)
    rc = shlex.quote(str(rcfile))
    cmd = f"{shlex.join(argv)} >> {shlex.quote(str(logfile))} 2>&1; echo $? > {rc}.tmp && mv {rc}.tmp {rc}"
    pr = subprocess.Popen(["setsid", "bash", "-c", cmd], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    Path(str(rcfile) + ".pid").write_text(str(pr.pid))


def bg_lost(rcfile):
    """True if the background op behind rcfile died without writing it (e.g. the controller restarted)."""
    if Path(rcfile).exists():
        return False
    try:
        pid = int(Path(str(rcfile) + ".pid").read_text())
    except Exception:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return not Path(rcfile).exists()
    except PermissionError:
        pass
    return False


def rsync_argv(h, src, dst, excludes=()):
    a = ["rsync", "-aH", "--partial", f"--bwlimit={h['bwlimit']}"]
    if h["ssh"] != "local":
        a += ["-e", shlex.join(ssh_argv(h))]
    return a + [f"--exclude={e}" for e in excludes] + [src, dst]


def start_staging(c, j, h):
    """Copy the job's workdir (and --sync-path paths) to a remote host in the background."""
    spec = json.loads(j["spec"])
    paths = ([spec["workdir"]] if spec.get("sync", True) else []) + spec.get("sync_paths", [])
    local = WORK / str(j["id"])
    local.mkdir(parents=True, exist_ok=True)
    # a directory is mirrored into the same path; a single file goes into its (created) parent directory
    isfile = {p: os.path.isfile(os.path.expanduser(p)) for p in paths}
    mk = " ".join(shlex.quote(os.path.dirname(remap(p, h).rstrip("/")) if isfile[p] else remap(p, h)) for p in paths)
    rc, _, err = sh(h, f"mkdir -p {mk}")
    if rc:
        return fail(c, j, f"staging mkdir failed: {err.strip()[-200:]}")
    steps = [shlex.join(rsync_argv(h, p, rsync_target(h, os.path.dirname(remap(p, h)) + "/"), spec.get("exclude", [])))
             if isfile[p] else
             shlex.join(rsync_argv(h, p.rstrip("/") + "/", rsync_target(h, remap(p, h).rstrip("/") + "/"),
                                   spec.get("exclude", []))) for p in paths]
    (local / "stage.sh").write_text("set -e\n" + "\n".join(steps) + "\n")
    bg(["bash", str(local / "stage.sh")], local / "stage.rc", local / "stage.log")
    upd(c, j["id"], state="staging", note="syncing workdir")


def stem_of(name):
    """A job name's stem: the trailing run of index suffixes stripped repeatedly (run-4 and run-12 share the stem run)."""
    while True:
        s = re.sub(r"[-_]?\d+[A-Za-z]*$", "", name)
        if s == name:
            return name
        name = s


def est_secs(c, name, h):
    """Runtime estimate in seconds for a job called `name` on Slurm host h, from this DB's finished jobs
    (state 'done', rc 0) whose name has the same stem: 1.5 x the max of the last 20 runtimes, rounded up to
    30 min, clamped to [1 h, max_time]. None with fewer than 2 samples."""
    if h.get("backend") != "slurm":
        return None
    runs = sorted(r["ended"] - r["started"] for r in c.execute(
        "select started, ended, name from jobs where state='done' and rc=0 and host=? "
        "and started is not null and ended is not null", (h["name"],))
        if stem_of(r["name"] or "") == stem_of(name or ""))
    if len(runs) < 2:
        return None
    est = -(-int(1.5 * max(runs[-20:])) // 1800) * 1800  # round up to 30 min
    return min(max(est, 3600), time_secs(h["max_time"]) if h.get("max_time") else 86400)  # capped by max_time too


def est_time(c, name, h):
    """A realistic Slurm -t value for such a job: the estimate, else the host's default_time."""
    s = est_secs(c, name, h)
    return f"{s // 3600}:{s % 3600 // 60:02d}:00" if s is not None else h.get("default_time")


def time_secs(t):
    """A Slurm time limit string -> seconds (minutes, minutes:seconds, hours:minutes:seconds, days-h[:mm[:ss]])."""
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d) * 86400
    p = [int(x) for x in t.split(":")]
    if len(p) == 1:  # a bare number is minutes, but days-h means hours
        return days + p[0] * (3600 if days else 60)
    p = [0] * (3 - len(p)) + p
    return days + p[0] * 3600 + p[1] * 60 + p[2]


def slurm_argv(c, j, h, jobdir, gpus):
    """The sbatch line launch() would run for job j on Slurm host h (kept separate so tests can build it).
    Transparent names: -J is the gsched job's own name, sanitized; --comment carries "gsched #<id> <client>".
    Every lookup of our Slurm jobs goes through the Slurm job id stored in the DB handle."""
    spec, jid = json.loads(j["spec"]), j["id"]
    g = {**gpus, "type": h.get("gres_types", {}).get(gpus["type"], gpus["type"])}  # gsched card type -> Slurm GRES
    gres = [f"gpu:{g['type']}:{g['n']}" if g["type"] != "gpu" else f"gpu:{g['n']}"]
    gres += [f"localdisk:{pool}:{gb}G" for pool, gb in local_disk_spec(spec, h)]  # node-local SSD/HDD scratch
    sname = re.sub(r"[^A-Za-z0-9._-]", "", j["name"] or "")[:48] or f"gsched-{jid}"
    t = spec.get("time") or est_time(c, j["name"], h) or h.get("default_time", "1-00:00:00")
    mt = h.get("max_time")
    if mt and time_secs(t) > time_secs(mt):  # nothing of ours sits on a Slurm host for longer than max_time
        t = mt
    args = ["sbatch", "--parsable", "-p", h["partition"], "-J", sname, "--comment", f"gsched #{jid} {client_of(j)}",
            "-o", f"{jobdir}/slurm.out", "--gres=" + ",".join(gres),
            "-c", str(spec.get("cpus", 4)), "-t", t]
    if spec.get("mem_gb"):
        args += [f"--mem={int(-(-spec['mem_gb'] // 1))}G"]
    if spec.get("slurm_exclude"):
        args += [f"--exclude={spec['slurm_exclude']}"]
    return args


def launch(c, j, h):
    spec = json.loads(j["spec"])
    jid = j["id"]
    jobdir = f"{h['jobroot']}/{jid}"
    gpus = json.loads(j["gpus"] or "[]")
    run, wrap = scripts(spec, h, jobdir, jid, gpus if isinstance(gpus, list) else [])
    q = shlex.quote(jobdir)
    payload = json.dumps({"run.sh": run, "wrap.sh": wrap})
    write = (f"mkdir -p {q} && rm -f {q}/rc {q}/pid {q}/start && python3 -c "
             + shlex.quote("import json,sys,os\nd=sys.argv[1]\nfor k,v in json.load(sys.stdin).items(): open(os.path.join(d,k),'w').write(v)")
             + f" {q}")
    if h["backend"] == "slurm":
        cmd = f"{write} && {shlex.join(slurm_argv(c, j, h, jobdir, gpus))} {q}/wrap.sh"
    else:
        cmd = f"{write} && cd {q} && (setsid nohup bash {q}/wrap.sh > /dev/null 2>&1 < /dev/null &) && echo started"
    rc, out, err = sh(h, cmd, stdin=payload)
    if rc:
        return fail(c, j, f"launch failed: {(err or out).strip()[-300:]}")
    handle = out.strip().splitlines()[-1].split(";")[0] if h["backend"] == "slurm" else ""
    upd(c, jid, state="running", jobdir=jobdir, handle=handle, started=time.time(), attempts=j["attempts"] + 1,
        note="submitted to slurm" if handle else "")
    log(f"job {jid} -> {h['name']} {gpus}" + (f" (slurm {handle})" if handle else ""))


def fail(c, j, note, state="failed"):
    upd(c, j["id"], state=state, ended=time.time(), note=note)
    log(f"job {j['id']} {state}: {note}")


def finish(c, j, h, rc):
    """Job ended remotely: fetch log + outputs in the background, then finalize."""
    spec = json.loads(j["spec"])
    local = WORK / str(j["id"])
    local.mkdir(parents=True, exist_ok=True)
    upd(c, j["id"], rc=rc, ended=time.time())
    if h["ssh"] == "local" and Path(j["jobdir"]) == local:
        return finalize(c, j["id"], rc)
    steps = [shlex.join(rsync_argv(h, rsync_target(h, j["jobdir"] + "/"), str(local) + "/", ["wrap.sh"]))]
    for o in spec.get("outputs", []):
        src = remap(os.path.join(spec["workdir"], o), h)
        dst = os.path.join(spec["workdir"], o)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        steps.append(shlex.join(rsync_argv(h, rsync_target(h, src), str(Path(dst).parent) + "/")))
    (local / "fetch.sh").write_text("set -e\n" + "\n".join(steps) + "\n")
    bg(["bash", str(local / "fetch.sh")], local / "fetch.rc", local / "fetch.log")
    upd(c, j["id"], state="fetching", note="fetching log/outputs")


def finalize(c, jid, rc, note=""):
    ended = c.execute("select ended from jobs where id=?", (jid,)).fetchone()[0] or time.time()
    upd(c, jid, state="done" if rc == 0 else "failed", ended=ended, note=note)
    log(f"job {jid} {'done' if rc == 0 else f'failed rc={rc}'}")


def requeue_or_fail(c, j, why):
    spec = json.loads(j["spec"])
    if j["attempts"] <= spec.get("retries", 0):
        upd(c, j["id"], state="queued", host=None, gpus=None, handle=None, started=None, note=f"requeued: {why}")
        log(f"job {j['id']} requeued: {why}")
    else:
        fail(c, j, why, state="failed")


# ---------------------------------------------------------------- reservations (optional)
# If people book GPUs somewhere else (a shared sheet, a chat bot, a web board), gsched can respect that:
# `reservations_cmd` in [defaults] runs once per pass on the controller and prints JSON
# {"<host>": {"<gpu index>": ["holder", ...]}}. A card held by anyone but the OS user is never idle for gsched.
# If the command fails, no card counts as reserved (and the failure is logged).

def reservations(d, hosts, quiet=False):
    """{host: {gpu index: [holders other than us]}} from reservations_cmd ({} per host when unset or failing)."""
    out = {n: {} for n in hosts}
    cmd = d.get("reservations_cmd")
    if not cmd:
        return out
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
        if r.returncode:
            raise ValueError(f"rc={r.returncode}: {r.stderr.strip()[-200:]}")
        for n, cards in json.loads(r.stdout).items():
            if n in out:
                out[n] = {int(i): [x for x in holders if x != USER] for i, holders in cards.items()}
    except (OSError, ValueError, TypeError, AttributeError, subprocess.TimeoutExpired) as e:
        if not quiet:
            log(f"reservations_cmd failed ({e}); no card counts as reserved")
    return out


# ---------------------------------------------------------------- one scheduling pass

def gather(c, d, hosts):
    jobs = c.execute(f"select * from jobs where state in {ACTIVE} order by id").fetchall()
    per = {n: {str(j["id"]): j["jobdir"] for j in jobs if j["host"] == n and j["state"] == "running"} for n in hosts}
    with cf.ThreadPoolExecutor(len(hosts)) as ex:
        res = dict(zip(hosts, ex.map(lambda n: probe(hosts[n], per[n]), hosts)))
    now = time.time()
    held = reservations(d, hosts)
    views = {}
    for n, h in hosts.items():
        p = res[n]
        if "error" in p:
            views[n] = None
            continue
        mine = [j for j in jobs if j["host"] == n and j["state"] in ("staging", "running")]
        if h["backend"] == "slurm":
            v = slurm_view(p)
            v["mine_gpus"] = sum(json.loads(j["gpus"])["n"] for j in mine)
            v["pending"] = sum(1 for j in mine if (v["squeue"].get(j["handle"] or "") or [None, None])[1] == "PENDING")
            v["client_gpus"] = {}
            for j in mine:
                v["client_gpus"][client_of(j)] = v["client_gpus"].get(client_of(j), 0) + json.loads(j["gpus"])["n"]
        else:
            row = c.execute("select data from hoststate where host=?", (n,)).fetchone()
            prev = json.loads(row["data"]) if row else {}
            v = {"gpus": node_view(h, p, d, prev, now, mine, held[n]), "mem_avail_gb": p["mem_avail_gb"],
                 "load1": p["load1"], "ncpu": p["ncpu"]}
            cards = {}  # distinct cards per client (packed --gpu-mem jobs share one)
            for j in mine:
                cards.setdefault(client_of(j), set()).update(json.loads(j["gpus"] or "[]"))
            v["client_cards"] = cards
            v["client_gpus"] = {k: len(x) for k, x in cards.items()}
        v["jobs"] = p["jobs"]
        views[n] = v
    return jobs, res, views


def save_idle(c, views):
    for n, v in views.items():
        if v and "gpus" in v:
            data = {str(g["index"]): g["idle_since"] for g in v["gpus"] if g["idle_since"] is not None}
            c.execute("insert or replace into hoststate values(?,?,?)", (n, json.dumps(data), time.time()))
    c.commit()


def tick(dry=False):
    c = db()
    try:
        return _tick(c, dry)
    finally:
        c.close()  # the daemon calls tick() forever; leaked connections ran it out of fds


def _tick(c, dry):
    d, hosts = load_config()
    jobs, res, views = gather(c, d, hosts)
    for n, v in views.items():
        if v is None:
            log(f"{n}: unreachable ({res[n]['error']})")
    if not dry:
        save_idle(c, views)
        # 0. jobs on a host that is gone from the config can't be tracked any more
        for j in jobs:
            if j["host"] and j["host"] not in hosts and j["state"] in ("staging", "running"):
                fail(c, j, f"host {j['host']} is no longer in {CONFIG}; check it by hand")
        jobs = [j for j in jobs if not j["host"] or j["host"] in hosts or j["state"] == "fetching"]
        # 1. running jobs
        for j in jobs:
            h, v = hosts.get(j["host"]), views.get(j["host"])
            if j["state"] != "running" or not v:
                continue
            st = v["jobs"].get(str(j["id"]), {})
            if h["backend"] == "slurm" and st.get("start") and st["start"] != j["started"]:
                upd(c, j["id"], started=st["start"], note="")  # sbatch time -> when the Slurm job really started
            if st.get("rc") is not None:
                finish(c, j, h, st["rc"])
            elif h["backend"] == "slurm":
                sq = v["squeue"].get(j["handle"])
                if sq is None and v["squeue_ok"] and time.time() - (j["started"] or 0) > 120:
                    requeue_or_fail(c, j, "slurm job vanished without an exit code")
                elif sq and sq[1] == "PENDING":
                    upd(c, j["id"], note=f"slurm pending: {sq[3]}")
            elif not st.get("alive") and time.time() - (j["started"] or 0) > 60:
                requeue_or_fail(c, j, "process gone without an exit code (reboot / OOM kill?)")
        # 2. background staging / fetching
        for j in jobs:
            local = WORK / str(j["id"])
            if j["state"] == "staging" and (local / "stage.rc").exists():
                if quiet_now(hosts[j["host"]]):
                    pass  # synced, but the host is in quiet hours: launch once they are over
                elif (local / "stage.rc").read_text().strip() == "0":
                    launch(c, c.execute("select * from jobs where id=?", (j["id"],)).fetchone(), hosts[j["host"]])
                else:
                    fail(c, j, f"staging failed, see {local}/stage.log")
            for op, state in (("stage", "staging"), ("fetch", "fetching")):
                if j["state"] == state and bg_lost(local / f"{op}.rc"):
                    log(f"job {j['id']}: {op} op died, restarting it")
                    bg(["bash", str(local / f"{op}.sh")], local / f"{op}.rc", local / f"{op}.log")
            if j["state"] == "fetching" and (local / "fetch.rc").exists():
                ok = (local / "fetch.rc").read_text().strip() == "0"
                finalize(c, j["id"], j["rc"], "" if ok else f"fetch failed, see {local}/fetch.log")
        quiet_signal(c, hosts)
        jobs, res, views = gather(c, d, hosts)
    # 3. place queued jobs, priority then FIFO
    plan, out = {}, []
    queued = [j for j in jobs if j["state"] == "queued"]
    queued.sort(key=lambda j: (-json.loads(j["spec"]).get("priority", 0), j["id"]))
    live = {n: v for n, v in views.items() if v}
    syncing = {j["host"] for j in jobs if j["state"] == "staging"}
    for j in queued:
        spec = {**json.loads(j["spec"]), "client": client_of(j)}
        r = place(spec, (time.time() - j["submitted"]) / 60, live, hosts, d, plan, syncing)
        if not r:
            continue
        n, g = r
        h = hosts[n]
        plan.setdefault(n, []).append({"gpus": g if isinstance(g, list) else [], "gpu_mem_gb": spec.get("gpu_mem_gb"),
                                       "mem_gb": spec.get("mem_gb", 0), "cpus": spec.get("cpus", 1), "client": spec["client"],
                                       **(g if isinstance(g, dict) else {})})
        out.append((j["id"], n, g))
        if needs_sync(spec, h):
            syncing.add(n)
        if dry:
            continue
        upd(c, j["id"], host=n, gpus=json.dumps(g), note="")
        j = c.execute("select * from jobs where id=?", (j["id"],)).fetchone()
        if needs_sync(spec, h):
            start_staging(c, j, h)
        else:
            launch(c, j, h)
    return out, views, hosts


# ---------------------------------------------------------------- CLI

def locked(fn, wait_msg=None):
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "tick.lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if wait_msg:
                print(wait_msg, file=sys.stderr, flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        return fn()


def parse_kv(items):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        out[k] = v
    return out


def cmd_submit(a):
    if a.file:
        specs = [json.loads(l) for l in Path(a.file).read_text().splitlines() if l.strip()]
    else:
        if not a.cmd:
            sys.exit("give a command after --, or --file jobs.jsonl")
        cmd = a.cmd[1:] if a.cmd[0] == "--" else a.cmd
        specs = [{"cmd": cmd[0] if len(cmd) == 1 else shlex.join(cmd)}]
    base = {"workdir": homeform(a.workdir), "gpus": a.gpus, "gpu_mem_gb": a.gpu_mem, "cpus": a.cpus,
            "mem_gb": a.mem, "hosts": a.hosts.split(",") if a.hosts else None, "fallback": a.fallback,
            "priority": a.priority, "retries": a.retries, "time": a.time, "env": parse_kv(a.env),
            "sync": not a.no_sync, "sync_paths": [homeform(p) for p in a.sync_path or []],
            "exclude": a.exclude or [], "outputs": [o.rstrip("/") for o in a.output or []], "setup_on": parse_kv(a.setup_on),
            "gpu_type": a.gpu_type, "name": a.name, "slurm_exclude": a.slurm_exclude, "client": a.client,
            "queue_ahead": a.queue_ahead or None, "local_disk": a.local_disk}
    c = db()
    _, hosts = load_config()
    for s in specs:
        spec = {**base, **{k: v for k, v in s.items() if v is not None}}
        spec = {k: v for k, v in spec.items() if v not in (None, [], {}, "")}
        spec["client"] = spec.get("client") or USER
        for k, lo in (("gpus", 1), ("cpus", 1), ("mem_gb", 0), ("gpu_mem_gb", 0.001), ("priority", None), ("retries", 0)):
            v = spec.get(k)
            if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool) or v != v
                                  or (lo is not None and v < lo) or (k in ("gpus", "cpus", "retries") and v != int(v))):
                sys.exit(f"bad {k} {v!r}")
        bad = [k for k in spec.get("env", {}) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)]
        if bad:
            sys.exit(f"bad environment variable name {bad[0]!r}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", spec["client"]):
            sys.exit(f"bad client name {spec['client']!r} (letters, digits, _ . -)")
        if spec.get("local_disk") and spec["local_disk"] != "none" and not re.fullmatch(
                r"[A-Za-z0-9_]+:\d+(,[A-Za-z0-9_]+:\d+)*", spec["local_disk"]):
            sys.exit(f"bad --local-disk {spec['local_disk']!r} (POOL:GiB[,POOL:GiB] or none)")
        for n in spec.get("hosts", []):
            if n not in hosts:
                sys.exit(f"unknown host {n}; known: {', '.join(hosts)}")
            if not hosts[n]["enabled"]:
                sys.exit(f"host {n} is disabled in {CONFIG} (enabled = false)")
        # warn (but still accept) when the given --time is way above what past jobs with the same name stem needed,
        # or beyond max_time on every allowed Slurm host (it would be clamped there; node hosts have no limit)
        if spec.get("time"):
            lim = time_secs(spec["time"])
            slurm_hosts = [hosts[n] for n in allowed_hosts(spec, hosts) if hosts[n]["backend"] == "slurm"]
            for hs in slurm_hosts:
                est = est_secs(c, spec.get("name") or spec["cmd"][:40], hs)
                if est is not None and lim > 3 * est:
                    print(f"gsched: warning: --time {spec['time']} is over 3x the {est // 3600}h{(est % 3600) // 60:02d}m "
                          f"estimated from past {hs['name']} jobs with the same name stem; a tighter limit queues friendlier "
                          "for other users", file=sys.stderr)
                    break
            if slurm_hosts and all(hs.get("max_time") and lim > time_secs(hs["max_time"]) for hs in slurm_hosts):
                mts = ", ".join(f"{hs['name']} {hs['max_time']}" for hs in slurm_hosts)
                print(f"gsched: warning: --time {spec['time']} exceeds max_time on every allowed Slurm host ({mts}); "
                      "launch will clamp it there (node hosts have no limit)", file=sys.stderr)
        cur = c.execute("insert into jobs(name, spec, state, submitted, client) values(?,?,?,?,?)",
                        (spec.get("name") or spec["cmd"][:40], json.dumps(spec), "queued", time.time(), spec["client"]))
        c.commit()
        print(cur.lastrowid)


def fmt_t(t):
    return time.strftime("%m-%d %H:%M", time.localtime(t)) if t else "-"


def dur(a, b):
    if not a:
        return "-"
    s = int((b or time.time()) - a)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def cmd_ls(a):
    c = db()
    q = "select * from jobs" + ("" if a.all else f" where (state in {ACTIVE} or ended > {time.time() - 86400})")
    rows = [j for j in c.execute(q + " order by id").fetchall() if not a.client or client_of(j) == a.client]
    print(f"{'id':>4} {'state':9} {'client':10} {'host':9} {'gpus':14} {'submitted':11} {'runtime':8} {'rc':>3}  name / note")
    for j in rows:
        g = j["gpus"] or ""
        print(f"{j['id']:>4} {j['state']:9} {client_of(j)[:10]:10} {j['host'] or '-':9} {g[:14]:14} {fmt_t(j['submitted']):11} "
              f"{dur(j['started'], j['ended']):8} {'' if j['rc'] is None else j['rc']:>3}  {j['name']}"
              + (f"  [{j['note']}]" if j["note"] else ""))


def cmd_status(a):
    d, hosts = load_config()
    c = db()
    _, res, views = gather(c, d, hosts)
    for n, v in views.items():
        h = hosts[n]
        caps = {k: x["max_gpus"][n] for k, x in d["clients"].items() if n in x["max_gpus"]}
        tag = (f"tier {h['tier']}" + (", opt-in fallback" if h["optin"] else "")
               + (f", max {h['max_gpus']} GPUs" if h["max_gpus"] < 1 << 30 else "")
               + (f" ({', '.join(f'{k} ≤{x}' for k, x in caps.items())})" if caps else "")
               + (", DISABLED (never placed)" if not h["enabled"] else "")
               + (f", quiet {h['quiet_hours']}" + (" (now: no new jobs, running ones paused)" if quiet_now(h) else "")
                  if h["quiet_hours"] else ""))
        if v is None:
            print(f"{n} ({tag}): UNREACHABLE {res[n]['error']}")
            continue
        if h["backend"] == "slurm":
            # Slurm's GRES type can hide variants (all A100s are "A100"); gpu_labels maps "node:type" -> real card
            label = lambda node, t: h.get("gpu_labels", {}).get(f"{node}:{t}", t)
            fr, tot = {}, {}
            for line in res[n]["nodes"].strip().splitlines():
                node = line.split()[0]
                for t, k in parse_gres(line.split()[1]).items():
                    tot[label(node, t)] = tot.get(label(node, t), 0) + k
            for node, f in v["free"].items():
                for t, k in f.items():
                    fr[label(node, t)] = fr.get(label(node, t), 0) + k
            per = v["client_gpus"]
            mine = (f"{a.client} {per.get(a.client, 0)}" if a.client
                    else f"{v['mine_gpus']}" + (f" ({', '.join(f'{k} {x}' for k, x in sorted(per.items()))})"
                                               if set(per) - {USER} else ""))
            print(f"{n} ({tag}, slurm): free {', '.join(f'{t}×{fr.get(t, 0)}/{k}' for t, k in sorted(tot.items()))}; "
                  f"mine {mine}; my slurm jobs {len(v['squeue'])}")
            cpu_models = h.get("cpu_models", {})
            for node, r in sorted(v["nodes"].items()):
                gp = ", ".join(f"{label(node, t)} {r['free'].get(t, 0)}/{k}" for t, k in sorted(r["gres"].items()))
                ld = f"load {r['load']:.0f}" if r.get("load") is not None else ""
                ram = f"{r['mem_total_gb'] - r['mem_unalloc_gb']:.0f}/{r['mem_total_gb']:.0f}G"
                print(f"  {node:<10} {r['state']:<6} CPU {r['cpu_alloc']:>2}/{r['cpu_total']:<2} {ld:<8} RAM {ram:>9}  "
                      f"{cpu_models.get(node, ''):<18}  GPU free {gp or '-'}"
                      + ("  disk free " + " ".join(f"{k} {r['disk_free'][k]:.0f}/{t:.0f}G"
                                                   for k, t in sorted(r.get("disk_total", {}).items()))
                         if r.get("disk_total") else ""))
            continue
        print(f"{n} ({tag}): RAM avail {v['mem_avail_gb']:.0f} GiB, load {v['load1']:.1f}/{v['ncpu']}")
        for g in v["gpus"]:
            jobs = [j for j in g["jobs"] if not a.client or client_of(j) == a.client]
            who = ("gsched " + ",".join(str(j["id"]) + (f"({client_of(j)})" if client_of(j) != USER else "") for j in jobs)
                   if jobs else "gsched (other clients)" if g["jobs"]
                   else f"foreign {','.join(g['foreign'])}" if g["foreign"]
                   else f"reserved {','.join(g['reserved'])}" if g["reserved"]
                   else "mine (not gsched)" if g["mine_other"] else "idle" if g["idle"]
                   else "settling" if g["idle_since"] else "busy")
            print(f"  GPU{g['index']}: {g['used'] / 1024:5.1f}/{g['total'] / 1024:.0f} GiB  util {g['util']:3d}%  {who}")


def cmd_plan(a):
    out, _, _ = locked(lambda: tick(dry=True))
    for jid, n, g in out:
        print(f"job {jid} -> {n} {g}")
    if not out:
        print("nothing would start now")


def get(c, jid):
    j = c.execute("select * from jobs where id=?", (jid,)).fetchone()
    if not j:
        sys.exit(f"no job {jid}")
    return j


def cmd_show(a):
    j = get(db(), a.id)
    print(json.dumps({**dict(j), "spec": json.loads(j["spec"])}, indent=2, default=str))


def cmd_logs(a):
    c = db()
    j = get(c, a.id)
    local = WORK / str(j["id"]) / "out.log"
    _, hosts = load_config()
    if j["state"] in ("done", "failed", "cancelled") and local.exists() and not a.follow:
        sys.stdout.write("".join(local.read_text(errors="replace").splitlines(True)[-a.n:]))
        return
    if not j["jobdir"]:
        sys.exit(f"job {j['id']} has not started ({j['state']})")
    if j["host"] not in hosts:
        sys.exit(f"job {j['id']} ran on {j['host']}, which is no longer in {CONFIG}")
    h = hosts[j["host"]]
    t = f"tail -n {a.n} {'-F' if a.follow else ''} {shlex.quote(j['jobdir'])}/out.log"
    argv = ["bash", "-c", t] if h["ssh"] == "local" else [*ssh_argv(h), h["ssh"], t]
    os.execvp(argv[0], argv)


def cmd_cancel(a):
    locked(lambda: cancel(a.ids), wait_msg="waiting for the running scheduler pass to finish...")


def cancel(ids):
    """Under tick.lock, so a concurrent tick can't launch, fetch or overwrite a job being cancelled."""
    c = db()
    _, hosts = load_config()
    for jid in ids:
        j = get(c, jid)
        if j["state"] not in ACTIVE:
            print(f"job {jid} already {j['state']}")
            continue
        op = {"staging": "stage", "fetching": "fetch"}.get(j["state"])
        if op:  # stop the background rsync (started with setsid: its pid is its process group)
            try:
                os.killpg(int((WORK / str(jid) / f"{op}.rc.pid").read_text()), 15)
            except (OSError, ValueError):
                pass
        h = hosts.get(j["host"])
        if j["state"] == "running" and h is None:
            print(f"job {jid}: host {j['host']} is no longer configured; stop the process there by hand")
        elif j["state"] == "running":
            if h["backend"] == "slurm" and j["handle"]:
                rc, _, err = sh(h, f"scancel {shlex.quote(j['handle'])}")
            else:
                q = shlex.quote(j["jobdir"])
                rc, _, err = sh(h, f"p=$(cat {q}/pid 2>/dev/null) || exit 0; grep -q wrap.sh /proc/$p/cmdline 2>/dev/null "
                                   f"|| exit 0; kill -TERM -- -$p && kill -CONT -- -$p")  # CONT: wake a quiet-hours pause
            if rc:
                print(f"job {jid}: could not stop it on {j['host']} ({(err or f'rc={rc}').strip()[-200:]}); "
                      "left as is, try again")
                continue
        upd(c, jid, state="cancelled", ended=time.time(), note="cancelled")
        print(f"job {jid} cancelled")


def cmd_retry(a):
    c = db()
    for jid in a.ids:
        j = get(c, jid)
        if j["state"] in ACTIVE:
            print(f"job {jid} is {j['state']}")
            continue
        upd(c, jid, state="queued", host=None, gpus=None, handle=None, rc=None, started=None, ended=None,
            attempts=0, submitted=time.time(), note="retried")
        print(f"job {jid} queued again")


def cmd_init(a):
    if CONFIG.exists() and not a.force:
        sys.exit(f"{CONFIG} exists (--force overwrites it)")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text((Path(__file__).parent / "hosts.example.toml").read_text())
    print(f"wrote {CONFIG}: edit the [hosts.*] blocks, then try `gsched status` and `gtop`")


def cmd_tick(a):
    locked(lambda: tick(dry=False))


def cmd_daemon(a):
    d, _ = load_config()
    log(f"gsched daemon: poll every {d.get('poll_seconds', 60)}s, config {CONFIG}")
    while True:
        try:
            locked(lambda: tick(dry=False))
        except Exception as e:
            log(f"tick error: {e!r}")
        time.sleep(load_config()[0].get("poll_seconds", 60))


def main():
    p = argparse.ArgumentParser(prog="gsched", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"gsched {__version__}")
    sp = p.add_subparsers(dest="sub", required=True)
    s = sp.add_parser("init", help="write an example config")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)
    s = sp.add_parser("submit", help="queue a job")
    s.add_argument("-n", "--name")
    s.add_argument("-C", "--workdir", default=".", help="run here (synced to remote hosts at the same path under their home)")
    s.add_argument("-g", "--gpus", type=int, default=1)
    s.add_argument("--gpu-mem", type=float, help="GiB of GPU memory needed; lets the job share a GPU with other such jobs")
    s.add_argument("--cpus", type=int, default=4)
    s.add_argument("--mem", type=float, default=0, help="GiB host RAM")
    s.add_argument("--hosts", help="comma list of allowed hosts (default: all non-opt-in)")
    s.add_argument("--fallback", action="store_true", help="also allow opt-in hosts (optin = true) after fallback_after_min")
    s.add_argument("--gpu-type", help="required card type: only hosts whose gpu_types list it (e.g. A100)")
    s.add_argument("-p", "--priority", type=int, default=0)
    s.add_argument("--retries", type=int, default=0, help="requeue after reboot/kill (use with --resume-able jobs)")
    s.add_argument("--time", help="slurm time limit, e.g. 2-00:00:00")
    s.add_argument("--queue-ahead", dest="queue_ahead", action="store_true",
                   help="slurm hosts with queue_ahead = N: when no card is free, sbatch anyway (pending, any allowed type) "
                        "so the job waits in Slurm's queue; at most N such pending jobs per host")
    s.add_argument("--local-disk", dest="local_disk",
                   help="slurm node-local scratch, POOL:GiB[,POOL:GiB] (e.g. ssd0:50); default = the host's local_disk; "
                        "'none' = off. Mounted at /scratch/<pool> = $GSCHED_LOCAL, TMPDIR/PYSCF_TMPDIR point there; "
                        "deleted when the job ends, so copy results out yourself")
    s.add_argument("--slurm-exclude", dest="slurm_exclude", help="slurm nodes to avoid, e.g. gn02 (sbatch --exclude)")
    s.add_argument("-e", "--env", action="append", metavar="K=V")
    s.add_argument("--no-sync", action="store_true", help="workdir already exists on every allowed host")
    s.add_argument("--sync-path", action="append", help="extra local path to copy to the host (e.g. a dataset)")
    s.add_argument("--exclude", action="append", help="rsync exclude pattern for syncing (e.g. runs/, .venv/)")
    s.add_argument("-o", "--output", action="append", help="path relative to workdir to copy back when done")
    s.add_argument("--setup-on", action="append", metavar="HOST=CMD", help="extra setup line on one host")
    s.add_argument("--client", help="who submits (default: the OS user); [clients.<name>] in hosts.toml caps its cards")
    s.add_argument("--file", help="JSONL of job specs (keys as above, plus cmd)")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_submit)
    s = sp.add_parser("ls")
    s.add_argument("-a", "--all", action="store_true")
    s.add_argument("--client", help="only this client's jobs")
    s.set_defaults(fn=cmd_ls)
    s = sp.add_parser("status")
    s.add_argument("--client", help="only this client's jobs")
    s.set_defaults(fn=cmd_status)
    sp.add_parser("plan").set_defaults(fn=cmd_plan)
    s = sp.add_parser("show")
    s.add_argument("id", type=int)
    s.set_defaults(fn=cmd_show)
    s = sp.add_parser("logs")
    s.add_argument("id", type=int)
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("-n", type=int, default=50)
    s.set_defaults(fn=cmd_logs)
    for name, fn in (("cancel", cmd_cancel), ("retry", cmd_retry)):
        s = sp.add_parser(name)
        s.add_argument("ids", type=int, nargs="+")
        s.set_defaults(fn=fn)
    sp.add_parser("tick").set_defaults(fn=cmd_tick)
    sp.add_parser("daemon").set_defaults(fn=cmd_daemon)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
