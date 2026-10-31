#!/usr/bin/env python3
"""csched: a small CPU job queue over ssh (node hosts + Slurm clusters), stdlib only.

Separate from gsched (GPUs): csched only hands out CPU cores and RAM. A job is one command run for N tasks
(--tasks N, or one task per line of --task-file); each task gets --cpus cores and --mem GiB. Every tick the
controller looks at the free cores and RAM on every allowed host and cuts the job's queued tasks into shards,
one per host:
  node hosts: the shard's tasks run under xargs -P <lanes> in a setsid process group;
  slurm hosts: one sbatch --array with one element per lane, each running its tasks in turn.
A task writes rc/<i> when it ends; tasks whose rc never appears (node died, time limit) are requeued (--retries).
A shard whose tasks all fail marks the host bad for that job (missing env?) and its tasks go elsewhere.

  csched init                           write an example config to ~/.config/gsched/csched.toml
  csched submit [opts] -- CMD...       queue a job (prints its id); {i} = $CSCHED_TASK, {arg} = "$CSCHED_ARG"
  csched ls [-a]                        jobs
  csched status                         free cores / RAM per host, our use
  csched show ID | tasks ID | logs ID [TASK] | cancel ID... | retry ID...
  csched plan                           what the next tick would start where (no side effects)
  csched tick | daemon                  one scheduling pass | loop (csched.service)
"""
import argparse, concurrent.futures as cf, fcntl, getpass, json, math, os, re, shlex, sqlite3, subprocess, sys, time
import tomllib
from pathlib import Path

from . import __version__

_xdg = lambda var, default: Path(os.environ.get(var) or Path.home() / default)
CONFIG = Path(os.environ.get("CSCHED_CONFIG") or _xdg("XDG_CONFIG_HOME", ".config") / "gsched/csched.toml")
STATE = Path(os.environ.get("CSCHED_STATE") or _xdg("XDG_STATE_HOME", ".local/state") / "csched")
WORK = Path(os.environ.get("CSCHED_WORK") or STATE / "jobs")  # fetched results, per-shard logs, args files
LOCAL_HOME = str(Path.home())
USER = getpass.getuser()
ACTIVE_SHARD = ("staging", "running", "fetching")
NFS_GRACE = 75  # seconds between a Slurm shard leaving squeue and fetching its results


# ---------------------------------------------------------------- config / db

def load_config():
    if not CONFIG.exists():
        sys.exit(f"no config at {CONFIG}: run `csched init`, or point CSCHED_CONFIG at your csched.toml")
    cfg = tomllib.loads(CONFIG.read_text())
    d = cfg.get("defaults", {})
    d.setdefault("poll_seconds", 30)
    d.setdefault("mem_per_cpu_gb", 2)
    d.setdefault("max_batch", 8)
    d.setdefault("retries", 2)
    hosts = {}
    for name, h in cfg["hosts"].items():
        h = {**h, "name": name}
        h.setdefault("backend", "node")
        h.setdefault("ssh", name)
        h.setdefault("ssh_opts", [])
        h.setdefault("pref", 9)
        h.setdefault("enabled", True)
        h.setdefault("max_cpus", 8)
        h.setdefault("reserve_cpus", 0)
        h.setdefault("mem_reserve_gb", 8)
        h.setdefault("quiet_hours", "")
        h.setdefault("home", LOCAL_HOME)
        h.setdefault("jobroot", h["home"] + "/scratch/csched/jobs")
        h.setdefault("setup", "")
        h.setdefault("default_time", "1-00:00:00")
        h.setdefault("local_disk", "")
        h.setdefault("node_frac", 1.0)          # slurm: our cores on one node ≤ this × its cores
        h.setdefault("cores_per_free_gpu", 0)   # slurm: idle cores left per idle GPU on a node (for GPU users)
        hosts[name] = h
    return d, hosts


def db():
    STATE.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(STATE / "csched.db", timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("pragma journal_mode=wal")
    c.executescript("""
    create table if not exists jobs(id integer primary key, name text, spec text, state text, submitted real,
        ended real, ntasks integer, bad_hosts text default '[]', note text default '');
    create table if not exists tasks(job integer, idx integer, state text, rc integer, shard integer, host text,
        attempts integer default 0, started real, ended real, primary key(job, idx));
    create table if not exists shards(id integer primary key, job integer, host text, tasks text, lanes integer,
        state text, jobdir text, handle text, started real, ended real, paused integer default 0, note text default '',
        fetch_tries integer default 0);
    create table if not exists stages(job integer, host text, state text, started real, tries integer default 0,
        primary key(job, host));
    create index if not exists tasks_state on tasks(job, state);
    """)
    for tab, col, typ in (("shards", "fetch_tries", "integer default 0"), ("stages", "tries", "integer default 0"),
                          ("shards", "node", "text")):  # columns live DBs may lack
        if col not in {r[1] for r in c.execute(f"pragma table_info({tab})")}:
            c.execute(f"alter table {tab} add column {col} {typ}")
    return c


def log(msg):
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def upd(c, table, rid, **kw):
    c.execute(f"update {table} set {', '.join(k + '=?' for k in kw)} where id=?", (*kw.values(), rid))
    c.commit()


# ---------------------------------------------------------------- remote execution (same conventions as gsched)

def ssh_argv(h):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={Path.home()}/.ssh/csched-%C", "-o", "ControlPersist=5m",
            *h["ssh_opts"]]


def sh(h, cmd, stdin=None, timeout=120):
    argv = ["bash", "-c", cmd] if h["ssh"] == "local" else [*ssh_argv(h), h["ssh"], cmd]
    try:
        r = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    return r.returncode, r.stdout, r.stderr


def homeform(path):
    """Absolute path, with symlink targets inside ~ (e.g. ~/data -> /mnt/big/me) folded back under ~, so they remap
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
    path = str(path)
    if path == LOCAL_HOME or path.startswith(LOCAL_HOME + "/"):
        return h["home"] + path[len(LOCAL_HOME):]
    return path


def rsync_target(h, path):
    return path if h["ssh"] == "local" else f"{h['ssh']}:{path}"


def rsync_argv(h, src, dst, excludes=()):
    a = ["rsync", "-aH", "--partial"]
    if h["ssh"] != "local":
        a += ["-e", shlex.join(ssh_argv(h))]
    return a + [f"--exclude={e}" for e in excludes] + [src, dst]


def bg(argv, rcfile, logfile):
    """Run argv locally in the background; its exit code lands in rcfile."""
    Path(rcfile).unlink(missing_ok=True)
    # rc goes to a temp file first: bg_state must never see a half-written rc
    cmd = (f"{shlex.join(argv)} >> {shlex.quote(str(logfile))} 2>&1; "
           f"echo $? > {shlex.quote(str(rcfile))}.tmp && mv {shlex.quote(str(rcfile))}.tmp {shlex.quote(str(rcfile))}")
    pr = subprocess.Popen(["setsid", "bash", "-c", cmd], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    Path(str(rcfile) + ".pid").write_text(str(pr.pid))


def bg_state(rcfile):
    """None while running, the exit code when done, -1 when the background op died without one."""
    p = Path(rcfile)
    if p.exists():
        t = p.read_text().strip()
        if not t:
            return None  # empty rc: still running (the write is atomic, so this is transient)
        try:
            return int(t)
        except ValueError:
            return 1
    try:
        os.kill(int(Path(str(rcfile) + ".pid").read_text()), 0)
    except (OSError, ValueError):
        return None if p.exists() else -1
    return None


def quiet_now(h, now=None):
    if not h.get("quiet_hours"):
        return False
    a, b = (int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t in h["quiet_hours"].split("-"))
    t = time.localtime(now)
    m = t.tm_hour * 60 + t.tm_min
    return a <= m < b if a <= b else m >= a or m < b


# ---------------------------------------------------------------- probing

def probe(h, shards):
    """One ssh round trip: host load / Slurm view, plus progress of our running shards there (shell only: some
    clusters still have old python3)."""
    parts = []
    if h["backend"] == "node":
        parts.append("echo ncpu $(nproc); echo load $(cut -d' ' -f1 /proc/loadavg); "
                     "awk '/MemAvailable/{print \"memavail\", $2}' /proc/meminfo")
    else:
        p = shlex.quote(h["partition"])
        parts.append(f"echo '#sinfo'; sinfo -N -h -p {p} -O 'NodeList:40,StateCompact:20,CPUsState:30,Memory:20,AllocMem:20,Gres:300,GresUsed:300'"
                     f"; echo '#squeue'; sq=$(squeue -u $(id -un) -h -r -o '%F|%K|%j|%T|%C|%R' 2>&1) && "
                     # squeue can print an error and still exit 0 (e.g. an unknown user): then it is not ok
                     f"case \"$sq\" in *'squeue: error'*) echo \"$sq\" >&2;; *) [ -n \"$sq\" ] && echo \"$sq\"; echo '#sqok';; esac"
                     f"; echo '#end'")
    for s in shards:
        d = shlex.quote(s["jobdir"])
        parts.append(f"n=$(ls {d}/rc 2>/dev/null | grep -vc tmp); a=dead; "
                     f"[ -f {d}/pid ] && kill -0 -- -$(cat {d}/pid) 2>/dev/null && a=alive; "  # whole process group (setsid)
                     f"[ -f {d}/finished ] && a=finished; echo shard {s['id']} $n $a")
    rc, out, err = sh(h, "; ".join(parts), timeout=60)
    if rc and not out:
        return {"ok": False, "err": (err or "").strip()[-200:]}
    v = {"ok": True, "shards": {}}
    sect = None
    for line in out.splitlines():
        if line.startswith("#"):
            sect = line[1:]
            v["squeue_ok"] = v.get("squeue_ok") or sect == "sqok"
            continue
        f = line.split()
        if sect == "sinfo" and len(f) >= 5:
            try:
                a, i, o, t = (int(x) for x in f[2].split("/"))
                v.setdefault("nodes", {})[f[0]] = {
                    "state": f[1], "up": any(x in f[1] for x in ("idle", "mix")) and "*" not in f[1]
                                         and not any(x in f[1] for x in ("drain", "down", "maint", "resv")),
                    "cpu_idle": i, "cpu_total": t, "mem_total_gb": int(f[3]) / 1024,
                    "mem_unalloc_gb": (int(f[3]) - int(f[4])) / 1024,
                    "gpus": gpu_count(f[5]) if len(f) > 5 else 0, "gpus_used": gpu_count(f[6]) if len(f) > 6 else 0}
            except ValueError:
                pass
        elif sect == "squeue":
            x = line.split("|")
            if len(x) >= 6 and x[2].startswith("csched-"):
                v.setdefault("squeue", []).append({"job": x[0], "task": x[1], "name": x[2], "state": x[3],
                                                   "cpus": int(x[4]) if x[4].isdigit() else 0, "reason": x[5]})
        elif f and f[0] == "shard" and len(f) == 4:
            v["shards"][int(f[1])] = {"rc": int(f[2]), "proc": f[3]}
        elif f and f[0] in ("ncpu", "load", "memavail") and len(f) == 2:
            v[f[0]] = float(f[1])
    v.setdefault("nodes", {})
    v.setdefault("squeue", [])
    if "memavail" in v:
        v["mem_avail_gb"] = v.pop("memavail") / 1024 / 1024
    if h["backend"] == "node" and not all(k in v for k in ("ncpu", "load", "mem_avail_gb")):
        return {"ok": False, "err": "incomplete probe: ncpu/load/memavail missing"}
    return v


GPU = re.compile(r"gpu:(?:[^:,()]+:)?(\d+)")


def gpu_count(gres):
    return sum(int(n) for n in GPU.findall(gres))


def gather(c, hosts):
    """Probe every enabled host in parallel. Our usage per host comes from the DB (node) or squeue (slurm)."""
    running = [dict(r) for r in c.execute(f"select * from shards where state in {ACTIVE_SHARD}")]
    views = {}
    with cf.ThreadPoolExecutor(8) as ex:
        fut = {n: ex.submit(probe, h, [s for s in running if s["host"] == n and s["state"] == "running"])
               for n, h in hosts.items() if h["enabled"] or any(s["host"] == n for s in running)}
        for n, f in fut.items():
            try:
                views[n] = f.result()
            except Exception as e:  # noqa: BLE001
                views[n] = {"ok": False, "err": str(e)[-200:]}
    for n, v in views.items():
        if not v["ok"]:
            continue
        mine = [s for s in running if s["host"] == n]
        if hosts[n]["backend"] == "node":
            v["ours_cpus"] = sum(s["lanes"] * spec_of(c, s["job"])["cpus"] for s in mine if not s["paused"])
            v["ours_mem_gb"] = sum(s["lanes"] * spec_of(c, s["job"])["mem_gb"] for s in mine)
        else:
            # squeue failed: we can't tell what runs, so place nothing there this tick
            v["ours_cpus"] = sum(q["cpus"] for q in v["squeue"]) if v.get("squeue_ok") else hosts[n]["max_cpus"]
            v["ours_pending"] = sum(1 for q in v["squeue"] if q["state"] == "PENDING")
            # staging shards are not in squeue yet but will be soon
            v["ours_cpus"] += sum(s["lanes"] * spec_of(c, s["job"])["cpus"] for s in mine if s["state"] == "staging")
            # per node: every shard is pinned to one node (-w), so its squeue entries count there
            where = {s["handle"]: s["node"] for s in mine if s["handle"]}
            on = {}
            for q in v["squeue"]:
                if where.get(q["job"]):
                    on[where[q["job"]]] = on.get(where[q["job"]], 0) + q["cpus"]
            for s in mine:
                if s["state"] == "staging" and s["node"]:
                    on[s["node"]] = on.get(s["node"], 0) + s["lanes"] * spec_of(c, s["job"])["cpus"]
            v["ours_node"] = on
    return views


_specs = {}


def spec_of(c, jid):
    if jid not in _specs:
        _specs[jid] = json.loads(c.execute("select spec from jobs where id=?", (jid,)).fetchone()[0])
    return _specs[jid]


# ---------------------------------------------------------------- capacity / placement

def slots(spec, h, v, taken):
    """How many more tasks of this job's shape (cpus, mem) the host can start now. taken: cpus/mem planned this pass."""
    if not v or not v.get("ok") or quiet_now(h):
        return 0, {}
    c, m = spec["cpus"], spec["mem_gb"]
    tc, tm = taken.get("cpus", 0), taken.get("mem", 0)
    if h["backend"] == "node":
        ours = v["ours_cpus"] + tc
        foreign = max(0.0, v["load"] - v["ours_cpus"])
        free_cpu = min(h["max_cpus"] - ours, v["ncpu"] - h["reserve_cpus"] - foreign - ours)
        free_mem = v["mem_avail_gb"] - h["mem_reserve_gb"] - tm
        return max(0, min(int(free_cpu // c), int(free_mem // m) if m else 10**6)), {}
    m = max(1, math.ceil(m))  # sbatch --mem is integer GiB; budget with the same effective value
    cap = int((h["max_cpus"] - v["ours_cpus"] - tc) // c)
    if cap <= 0:
        return 0, {}
    excl = {x.strip() for x in (spec.get("slurm_exclude") or "").split(",") if x.strip()}
    used = taken.get("nodes", {})
    n, per = 0, {}
    for node, r in v["nodes"].items():
        if not r["up"] or node in excl:
            continue
        uc, um = used.get(node, (0, 0))
        cpu = node_room(h, v, node) - uc
        k = min(int(cpu // c), int((r["mem_unalloc_gb"] - um) // m) if m else 10**6)
        if k > 0:
            per[node] = k
            n += k
    n = min(n, cap)
    return max(0, n), per


def node_cap(h, r):
    return int(r["cpu_total"] * h["node_frac"])


def node_room(h, v, node):
    """Cores we may still take on one Slurm node: its idle cores minus those kept for its idle GPUs, and at most
    node_frac of its cores in total for us."""
    r = v["nodes"][node]
    keep = h["cores_per_free_gpu"] * max(0, r.get("gpus", 0) - r.get("gpus_used", 0))
    return max(0, min(r["cpu_idle"] - keep, node_cap(h, r) - v.get("ours_node", {}).get(node, 0)))


def allowed(spec, hosts, bad):
    names = spec.get("hosts") or [n for n, h in hosts.items() if h["enabled"]]
    return sorted((n for n in names if n in hosts and hosts[n]["enabled"] and n not in bad), key=lambda n: hosts[n]["pref"])


def plan(c, d, hosts, views):
    """Shards to start: [(job row, host, [task idx], lanes, node)]. Jobs by priority then FIFO; each job spreads over
    every allowed host with room, preferred hosts first; on Slurm hosts one shard per node (most room first)."""
    out, taken = [], {n: {} for n in hosts}
    jobs = c.execute("select * from jobs where state in ('queued','running')").fetchall()
    jobs = sorted(jobs, key=lambda j: (-json.loads(j["spec"]).get("priority", 0), j["id"]))
    for j in jobs:
        spec = json.loads(j["spec"])
        queued = [r[0] for r in c.execute("select idx from tasks where job=? and state='queued' order by idx", (j["id"],))]
        if not queued:
            continue
        bad = set(json.loads(j["bad_hosts"] or "[]"))
        if spec.get("max_cpus"):  # per-job concurrency cap
            run = c.execute(f"select coalesce(sum(lanes), 0) from shards where job=? and state in {ACTIVE_SHARD}",
                            (j["id"],)).fetchone()[0]
            limit = max(0, spec["max_cpus"] // spec["cpus"] - run)
        else:
            limit = 10**9
        hs = [(n, *slots(spec, hosts[n], views.get(n), taken[n])) for n in allowed(spec, hosts, bad)]
        hs = [(n, k, per) for n, k, per in hs if k > 0]
        total = min(sum(k for _, k, _ in hs), limit)
        if not total:
            continue
        batch = spec.get("batch") or max(1, min(d["max_batch"], math.ceil(len(queued) / total)))
        for n, k, per in hs:
            budget = k  # the host-wide cap (max_cpus) on top of the per-node room
            for node, kk in sorted(per.items(), key=lambda x: (-x[1], x[0])) if per else [(None, k)]:
                if not queued or limit <= 0 or budget <= 0:
                    break
                lanes = min(kk, budget, limit, math.ceil(len(queued) / batch))
                take, queued = queued[:lanes * batch], queued[lanes * batch:]
                lanes = min(lanes, len(take))
                limit -= lanes
                budget -= lanes
                out.append((j, n, take, lanes, node))
                t = taken[n]
                t["cpus"] = t.get("cpus", 0) + lanes * spec["cpus"]
                t["mem"] = t.get("mem", 0) + lanes * spec["mem_gb"]
                if node:  # book the lanes on the node so the next job sees less room there
                    uc, um = t.setdefault("nodes", {}).get(node, (0, 0))
                    t["nodes"][node] = (uc + lanes * spec["cpus"], um + lanes * spec["mem_gb"])
    return out


# ---------------------------------------------------------------- shard scripts

def local_disk(spec, h):
    v = spec.get("local_disk") or h.get("local_disk") or "none"
    if h["backend"] != "slurm" or v == "none":
        return []
    return [(p, int(g)) for p, g in (x.split(":") for x in v.split(","))]


def scripts(spec, h, jobdir, jid, sid, ntasks, lanes):
    wd = remap(spec["workdir"], h)
    q = shlex.quote(jobdir)
    env = {"CSCHED_JOB": str(jid), "CSCHED_SHARD": str(sid), "CSCHED_HOST": h["name"], "CSCHED_JOBDIR": jobdir,
           "CSCHED_CPUS": str(spec["cpus"])}
    env |= {k: str(spec["cpus"]) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                           "NUMEXPR_NUM_THREADS")}
    env |= spec.get("env", {})
    local = ""
    for i, (pool, _) in enumerate(local_disk(spec, h)):
        dd = f"/scratch/{pool}"
        local += f"export CSCHED_LOCAL_{pool.upper()}={dd}\n"
        if i == 0:
            tmp = [k for k in ("TMPDIR", "PYSCF_TMPDIR", "GAUSS_SCRDIR", "QCSCRATCH", "PSI_SCRATCH") if k not in spec.get("env", {})]
            local += (f"if mkdir -p {dd}/tmp 2>/dev/null; then export CSCHED_LOCAL={dd}\n"
                      + "".join(f"  export {k}={dd}/tmp\n" for k in tmp)
                      + f"else echo 'csched: {dd} not writable, using defaults' >&2; fi\n")
    sub = lambda s: (s.replace("{home}", h["home"]).replace("{workdir}", wd).replace("{host}", h["name"])
                     .replace("{i}", "${CSCHED_TASK}").replace("{arg}", '"${CSCHED_ARG}"'))
    setup = "\n".join(sub(s) for s in (h["setup"], spec.get("setup_on", {}).get(h["name"], "")) if s)
    run = ("#!/bin/bash\n" + "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in env.items()) + local
           + f"{setup}\ncd {shlex.quote(wd)} || exit 97\n{sub(spec['cmd'])}\n")
    # task.sh K: the K-th task of this shard (line K+1 of tasks.txt / args.txt); skipped when its rc exists (resume)
    task = (f"#!/bin/bash\nK=$1\ni=$(sed -n \"$((K+1))p\" {q}/tasks.txt)\n[ -n \"$i\" ] || exit 0\n"
            f"[ -e {q}/rc/$i ] && exit 0\n"
            f"export CSCHED_TASK=$i CSCHED_NTASKS={spec['ntasks']}\n"
            f"export CSCHED_ARG=\"$(sed -n \"$((K+1))p\" {q}/args.txt 2>/dev/null)\"\n"
            f"date +%s > {q}/start/$i\n"
            f"bash -l {q}/run.sh > {q}/log/$i.log 2>&1 < /dev/null\n"
            f"echo $? > {q}/rc/$i.tmp && mv {q}/rc/$i.tmp {q}/rc/$i\n")
    if h["backend"] == "node":
        main = (f"#!/bin/bash\necho $$ > {q}/pid\n"
                f"seq 0 {ntasks - 1} | xargs -P {lanes} -n 1 bash {q}/task.sh\n"
                f"echo done > {q}/finished\n")
    else:  # array element L runs tasks L, L+lanes, L+2*lanes, ...
        main = (f"#!/bin/bash\nfor ((K=SLURM_ARRAY_TASK_ID; K<{ntasks}; K+={lanes})); do bash {q}/task.sh $K; done\n")
    return {"run.sh": run, "task.sh": task, "main.sh": main}


# ---------------------------------------------------------------- lifecycle

def needs_sync(spec, h):
    return h["ssh"] != "local"


def start_shard(c, d, hosts, j, host, idxs, lanes, node=None):
    spec = json.loads(j["spec"])
    h = hosts[host]
    now = time.time()
    cur = c.execute("insert into shards(job, host, tasks, lanes, state, started, node) values(?,?,?,?,?,?,?)",
                    (j["id"], host, json.dumps(idxs), lanes, "staging", now, node))
    sid = cur.lastrowid
    jobdir = f"{h['jobroot']}/{j['id']}/{sid}"
    c.executemany("update tasks set state='assigned', shard=?, host=?, started=? where job=? and idx=?",
                  [(sid, host, now, j["id"], i) for i in idxs])
    c.execute("update shards set jobdir=? where id=?", (jobdir, sid))
    if j["state"] == "queued":
        c.execute("update jobs set state='running' where id=?", (j["id"],))
    c.commit()
    if needs_sync(spec, h) and spec.get("sync", True):
        st = c.execute("select state from stages where job=? and host=?", (j["id"], host)).fetchone()
        if not st:
            start_stage(c, spec, j["id"], h)
        elif st["state"] == "failed":
            c.execute("delete from stages where job=? and host=?", (j["id"], host))
            start_stage(c, spec, j["id"], h)
    log(f"job {j['id']} shard {sid} -> {host}{':' + node if node else ''}: {len(idxs)} tasks, {lanes} lanes")


def start_stage(c, spec, jid, h):
    """Copy the workdir (and --sync-path paths) to a host once per job."""
    paths = [spec["workdir"]] + spec.get("sync_paths", [])
    local = WORK / str(jid)
    local.mkdir(parents=True, exist_ok=True)
    isfile = {p: os.path.isfile(os.path.expanduser(p)) for p in paths}
    mk = " ".join(shlex.quote(os.path.dirname(remap(p, h).rstrip("/")) if isfile[p] else remap(p, h)) for p in paths)
    steps = [shlex.join(["mkdir", "-p", *shlex.split(mk)]) if h["ssh"] == "local"
             else shlex.join([*ssh_argv(h), h["ssh"], f"mkdir -p {mk}"])]
    steps += [shlex.join(rsync_argv(h, p, rsync_target(h, os.path.dirname(remap(p, h)) + "/"), spec.get("exclude", [])))
              if isfile[p] else
              shlex.join(rsync_argv(h, p.rstrip("/") + "/", rsync_target(h, remap(p, h).rstrip("/") + "/"),
                                    spec.get("exclude", []))) for p in paths]
    f = local / f"stage-{h['name']}"  # string append: with_suffix would mangle names containing dots
    Path(str(f) + ".sh").write_text("set -e\n" + "\n".join(steps) + "\n")
    bg(["bash", str(f) + ".sh"], str(f) + ".rc", str(f) + ".log")
    c.execute("insert into stages(job, host, state, started, tries) values(?,?,?,?,0) "
              "on conflict(job, host) do update set state='running', started=?",  # keep tries across retries
              (jid, h["name"], "running", time.time(), time.time()))
    c.commit()


def stage_state(c, jid, h, spec):
    if not needs_sync(spec, h) or not spec.get("sync", True):
        return "done"
    st = c.execute("select * from stages where job=? and host=?", (jid, h["name"])).fetchone()
    if not st:
        start_stage(c, spec, jid, h)  # first shard to need this host: start the copy now
        return "running"
    if st["state"] != "running":
        return st["state"]
    rc = bg_state(WORK / str(jid) / f"stage-{h['name']}.rc")
    if rc is None:
        return "running"
    if rc == 0:
        c.execute("update stages set state='done' where job=? and host=?", (jid, h["name"]))
        c.commit()
        return "done"
    tries = (st["tries"] or 0) + 1
    if tries <= 3:  # retry the copy a few times; only a later failure marks the host bad
        log(f"job {jid}: stage to {h['name']} failed (rc {rc}), retry {tries}/3")
        c.execute("update stages set state='running', tries=? where job=? and host=?", (tries, jid, h["name"]))
        c.commit()
        start_stage(c, spec, jid, h)
        return "running"
    c.execute("update stages set state='failed' where job=? and host=?", (jid, h["name"]))
    c.commit()
    return "failed"


def launch(c, s, h):
    spec = spec_of(c, s["job"])
    idxs = json.loads(s["tasks"])
    files = scripts(spec, h, s["jobdir"], s["job"], s["id"], len(idxs), s["lanes"])
    files["tasks.txt"] = "".join(f"{i}\n" for i in idxs)
    if spec.get("args_file"):
        args = Path(spec["args_file"]).read_text().splitlines()
        files["args.txt"] = "".join(args[i] + "\n" for i in idxs)
    q = shlex.quote(s["jobdir"])
    write = (f"mkdir -p {q}/rc {q}/log {q}/start && python3 -c "
             + shlex.quote("import json,sys,os\nd=sys.argv[1]\nfor k,v in json.load(sys.stdin).items(): open(os.path.join(d,k),'w').write(v)")
             + f" {q}")
    if h["backend"] == "slurm":
        args = ["sbatch", "--parsable", "-p", h["partition"], "-J", f"csched-{s['job']}-{s['id']}",
                "-o", f"{s['jobdir']}/slurm-%a.out", f"--array=0-{s['lanes'] - 1}", "-c", str(spec["cpus"]),
                f"--mem={max(1, math.ceil(spec['mem_gb']))}G", "-t", spec.get("time") or h["default_time"]]
        ld = local_disk(spec, h)
        if ld:
            args.append("--gres=" + ",".join(f"localdisk:{p}:{g}G" for p, g in ld))
        if s["node"]:
            args.append(f"--nodelist={s['node']}")  # placed per node by plan()
        elif spec.get("slurm_exclude"):
            args.append(f"--exclude={spec['slurm_exclude']}")
        cmd = f"{write} && {shlex.join(args)} {q}/main.sh"
    else:
        cmd = (f"{write} && cd {q} && (setsid nohup bash {q}/main.sh > {q}/main.log 2>&1 < /dev/null &)"
               f" && for i in $(seq 20); do [ -f pid ] && break; sleep 0.2; done && echo started")
    rc, out, err = sh(h, cmd, stdin=json.dumps(files))
    if rc in (124, 255):
        # timeout / ssh died: the sbatch or main.sh may have started anyway; look before requeueing
        if h["backend"] == "slurm":
            rc2, out2, _ = sh(h, f"squeue -u $(id -un) -h -n csched-{s['job']}-{s['id']} -o %F")
            if out2.strip():
                handle = out2.strip().splitlines()[-1].split(";")[0]
                upd(c, "shards", s["id"], state="running", handle=handle, started=time.time())
                return log(f"shard {s['id']} running on {h['name']} (slurm {handle}, recovered after rc {rc})")
        elif sh(h, f"test -f {shlex.quote(s['jobdir'])}/pid")[0] == 0:
            upd(c, "shards", s["id"], state="running", started=time.time())
            return log(f"shard {s['id']} running on {h['name']} (recovered after rc {rc})")
    if rc:
        note = f"launch failed: {(err or out).strip()[-300:]}"
        log(f"shard {s['id']} {note}")
        return shard_failed(c, s, note, host_bad=False)
    handle = out.strip().splitlines()[-1].split(";")[0] if h["backend"] == "slurm" else ""
    upd(c, "shards", s["id"], state="running", handle=handle, started=time.time())
    log(f"shard {s['id']} running on {h['name']}" + (f" (slurm {handle})" if handle else ""))


def shard_failed(c, s, note, host_bad):
    """Shard could not run at all: its tasks go back to the queue (no attempt counted)."""
    c.execute("update tasks set state='queued', shard=null, host=null where shard=? and state='assigned'", (s["id"],))
    if host_bad:
        mark_bad(c, s["job"], s["host"])
    c.execute("update shards set state='failed', ended=?, note=? where id=?", (time.time(), note, s["id"]))
    c.commit()


def mark_bad(c, jid, host):
    bad = set(json.loads(c.execute("select bad_hosts from jobs where id=?", (jid,)).fetchone()[0] or "[]"))
    bad.add(host)
    c.execute("update jobs set bad_hosts=? where id=?", (json.dumps(sorted(bad)), jid))
    log(f"job {jid}: host {host} marked bad (a whole shard failed)")


def start_fetch(c, s, h):
    spec = spec_of(c, s["job"])
    local = WORK / str(s["job"]) / str(s["id"])
    local.mkdir(parents=True, exist_ok=True)
    if h["ssh"] == "local" and Path(s["jobdir"]) == local:
        steps = []
    else:
        steps = [shlex.join(rsync_argv(h, rsync_target(h, s["jobdir"] + "/"), str(local) + "/"))]
    if needs_sync(spec, h):
        for o in spec.get("outputs", []):
            src = remap(os.path.join(spec["workdir"], o), h)
            dst = os.path.join(spec["workdir"], o)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            # --ignore-missing-args: a shard that died or was cancelled early may not have created the output yet
            steps.append(shlex.join(rsync_argv(h, rsync_target(h, src), str(Path(dst).parent) + "/")[:-2]
                                    + ["--ignore-missing-args", rsync_target(h, src), str(Path(dst).parent) + "/"]))
    (local / "fetch.sh").write_text("set -e\n" + "\n".join(steps) + "\n")
    bg(["bash", str(local / "fetch.sh")], local / "fetch.rc", local / "fetch.log")
    upd(c, "shards", s["id"], state="fetching")


def settle(c, d, s, cancelled=False, extra=""):
    """Fetch finished: read rc/<i> for every task of the shard; missing ones are requeued (or failed past retries)."""
    local = WORK / str(s["job"]) / str(s["id"])
    spec = spec_of(c, s["job"])
    idxs = json.loads(s["tasks"])
    now = time.time()
    ok = bad = lost = 0
    for i in idxs:
        f = local / "rc" / str(i)
        t = c.execute("select * from tasks where job=? and idx=?", (s["job"], i)).fetchone()
        if t["state"] != "assigned" or t["shard"] != s["id"]:
            continue
        if f.exists():
            try:
                rc = int(f.read_text().strip())
            except ValueError:
                rc = 1
            if rc == 0:
                ok += 1
            else:
                bad += 1
            c.execute("update tasks set state=?, rc=?, ended=?, attempts=attempts+1 where job=? and idx=?",
                      ("done" if rc == 0 else "failed", rc, now, s["job"], i))
        elif cancelled:
            c.execute("update tasks set state='cancelled', ended=? where job=? and idx=?", (now, s["job"], i))
        else:
            lost += 1
            retry = t["attempts"] < spec.get("retries", d["retries"])
            c.execute("update tasks set state=?, shard=?, host=?, attempts=attempts+1, ended=? where job=? and idx=?",
                      ("queued" if retry else "failed", None if retry else s["id"], None if retry else s["host"],
                       None if retry else now, s["job"], i))
    note = f"{ok} ok, {bad} failed, {lost} lost" + extra
    rcs = {int(f.read_text().strip() or 1) for f in (local / "rc").glob("[0-9]*") if f.read_text().strip().lstrip("-").isdigit()}
    elsewhere = c.execute("select count(*) from tasks where job=? and state='done' and host!=?", (s["job"], s["host"])).fetchone()[0]
    if bad and not ok and not cancelled and (rcs <= {97, 126, 127} or elsewhere):
        # whole shard failed with cd / command-not-found codes, or the same job works on other hosts: this host lacks
        # the env. Retry those tasks elsewhere if another host is allowed.
        mark_bad(c, s["job"], s["host"])
        _, hosts = load_config()
        j = c.execute("select * from jobs where id=?", (s["job"],)).fetchone()
        if allowed(spec, hosts, set(json.loads(j["bad_hosts"]))):
            c.execute("update tasks set state='queued', rc=null, shard=null, host=null, ended=null, attempts=attempts-1 "
                      "where shard=? and state='failed'", (s["id"],))
            note += "; host marked bad, failed tasks requeued"
    c.execute("update shards set state=?, ended=?, note=? where id=?",
              ("cancelled" if cancelled else "done", now, note, s["id"]))
    c.commit()
    log(f"shard {s['id']} (job {s['job']}, {s['host']}) settled: {note}")
    finish_job(c, s["job"])


def finish_job(c, jid):
    n = dict(c.execute("select state, count(*) from tasks where job=? group by state", (jid,)).fetchall())
    if n.get("queued") or n.get("assigned"):
        return
    if c.execute(f"select count(*) from shards where job=? and state in {ACTIVE_SHARD}", (jid,)).fetchone()[0]:
        return
    state = "cancelled" if n.get("cancelled") else "failed" if n.get("failed") else "done"
    c.execute("update jobs set state=?, ended=? where id=? and state in ('queued','running')", (state, time.time(), jid))
    c.commit()
    log(f"job {jid} {state}: {n}")


def quiet_signal(c, hosts):
    """Node hosts in quiet hours: SIGSTOP our running shards' process groups; SIGCONT after."""
    for s in c.execute("select * from shards where state='running'").fetchall():
        h = hosts.get(s["host"])
        if not h or h["backend"] != "node":
            continue
        want = 1 if quiet_now(h) else 0
        if want != s["paused"]:
            sig = "STOP" if want else "CONT"
            sh(h, f"kill -{sig} -- -$(cat {shlex.quote(s['jobdir'])}/pid)")
            upd(c, "shards", s["id"], paused=want)
            log(f"shard {s['id']} on {h['name']}: SIG{sig} (quiet hours)")


# ---------------------------------------------------------------- tick

def tick(dry=False):
    d, hosts = load_config()
    c = db()
    views = gather(c, hosts)
    if dry:
        return plan(c, d, hosts, views), views
    quiet_signal(c, hosts)
    # running shards: done when the processes are gone (node) or squeue lists none of its elements (slurm)
    for s in c.execute("select * from shards where state='running'").fetchall():
        h, v = hosts.get(s["host"]), views.get(s["host"])
        if not h or not v or not v.get("ok"):
            continue
        p = v["shards"].get(s["id"])
        if p is None:
            continue
        if h["backend"] == "node":
            ended = p["proc"] in ("finished", "dead")
        else:
            ended = v.get("squeue_ok") and not any(q["job"] == s["handle"] for q in v["squeue"])
        if ended and h["backend"] == "slurm":
            # rc files are written on the compute nodes; the login node's NFS directory cache can hide the last ones
            # for up to ~60 s, so fetch only after a grace period
            if not s["ended"]:
                upd(c, "shards", s["id"], ended=time.time())
            elif time.time() - s["ended"] >= NFS_GRACE:
                start_fetch(c, s, h)
        elif ended:
            start_fetch(c, s, h)
        elif h["backend"] == "slurm" and s["ended"] and v.get("squeue_ok"):
            upd(c, "shards", s["id"], ended=None)  # back in squeue (requeued?): restart the grace clock
    for s in c.execute("select * from shards where state='fetching'").fetchall():
        rc = bg_state(WORK / str(s["job"]) / str(s["id"]) / "fetch.rc")
        if rc is None:
            continue
        if rc != 0 and s["fetch_tries"] < 3 and hosts.get(s["host"]):
            log(f"shard {s['id']} fetch rc {rc}; retry {s['fetch_tries'] + 1}/3")
            start_fetch(c, s, hosts[s["host"]])
            upd(c, "shards", s["id"], fetch_tries=s["fetch_tries"] + 1)
            continue
        settle(c, d, s, cancelled=s["note"] == "cancelling", extra="; fetch failed" if rc != 0 else "")
    for s in c.execute("select * from shards where state='staging'").fetchall():
        h = hosts.get(s["host"])
        if not h or not h["enabled"]:
            shard_failed(c, s, f"host {s['host']} disabled or unknown", host_bad=False)
            continue
        st = stage_state(c, s["job"], h, spec_of(c, s["job"]))
        if st == "done":
            if h["backend"] == "node" and quiet_now(h):
                continue  # quiet hours: the shard launches when they end
            launch(c, s, h)
        elif st == "failed":
            shard_failed(c, s, "staging failed", host_bad=True)
    views = gather(c, hosts)  # fresh numbers after launches
    for j, host, idxs, lanes, node in plan(c, d, hosts, views):
        start_shard(c, d, hosts, j, host, idxs, lanes, node)
    for (jid,) in c.execute("select id from jobs where state='running'").fetchall():
        finish_job(c, jid)
    return None, views


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


# ---------------------------------------------------------------- CLI

def parse_kv(items):
    out = {}
    for s in items or []:
        k, _, v = s.partition("=")
        out[k] = v
    return out


def cmd_submit(a):
    d, hosts = load_config()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("give a command after --")
    if a.cpus < 1:
        sys.exit("--cpus must be >= 1")
    if a.mem < 0:
        sys.exit("--mem must be >= 0")
    if a.batch is not None and a.batch < 1:
        sys.exit("--batch must be >= 1")
    if a.max_cpus is not None and a.max_cpus < 1:
        sys.exit("--max-cpus must be >= 1")
    cmd = cmd[0] if len(cmd) == 1 else shlex.join(cmd)
    if a.task_file:
        lines = Path(a.task_file).read_text().splitlines()
        ntasks = len(lines)
    else:
        ntasks = a.tasks
    if ntasks < 1:
        sys.exit("no tasks")
    if a.hosts:
        for n in a.hosts.split(","):
            if n not in hosts:
                sys.exit(f"unknown host {n}; known: {', '.join(hosts)}")
            if not hosts[n]["enabled"]:
                sys.exit(f"host {n} is disabled in {CONFIG}")
    if a.local_disk and a.local_disk != "none" and not re.fullmatch(r"[A-Za-z0-9_]+:\d+(,[A-Za-z0-9_]+:\d+)*", a.local_disk):
        sys.exit(f"bad --local-disk {a.local_disk!r} (POOL:GiB[,POOL:GiB] or none)")
    spec = {"cmd": cmd, "workdir": homeform(a.workdir), "cpus": a.cpus,
            "mem_gb": a.mem if a.mem else a.cpus * d["mem_per_cpu_gb"], "ntasks": ntasks,
            "hosts": a.hosts.split(",") if a.hosts else None, "priority": a.priority, "time": a.time,
            "env": parse_kv(a.env), "sync": not a.no_sync, "sync_paths": [homeform(p) for p in a.sync_path or []],
            "exclude": a.exclude or [], "outputs": [o.rstrip("/") for o in a.output or []],
            "setup_on": parse_kv(a.setup_on), "batch": a.batch, "max_cpus": a.max_cpus, "retries": a.retries,
            "local_disk": a.local_disk, "slurm_exclude": a.slurm_exclude, "name": a.name}
    spec = {k: v for k, v in spec.items() if v not in (None, [], {}, "")}
    c = db()
    cur = c.execute("insert into jobs(name, spec, state, submitted, ntasks) values(?,?,?,?,?)",
                    (a.name or cmd[:40], json.dumps(spec), "queued", time.time(), ntasks))
    jid = cur.lastrowid
    if a.task_file:
        WORK.joinpath(str(jid)).mkdir(parents=True, exist_ok=True)
        af = WORK / str(jid) / "args.txt"
        af.write_text("".join(x + "\n" for x in lines))
        spec["args_file"] = str(af)
        c.execute("update jobs set spec=? where id=?", (json.dumps(spec), jid))
    c.executemany("insert into tasks(job, idx, state) values(?,?,'queued')", [(jid, i) for i in range(ntasks)])
    c.commit()
    print(jid)


def fmt_t(t):
    return time.strftime("%m-%d %H:%M", time.localtime(t)) if t else "-"


def dur(a, b):
    if not a:
        return "-"
    s = int((b or time.time()) - a)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def counts(c, jid):
    return dict(c.execute("select state, count(*) from tasks where job=? group by state", (jid,)).fetchall())


def job_rows(c, all_=False):
    q = "select * from jobs" + ("" if all_ else f" where state in ('queued','running') or ended > {time.time() - 86400}")
    return c.execute(q + " order by id").fetchall()


def cmd_ls(a):
    c = db()
    print(f"{'id':>4} {'state':9} {'tasks':>6} {'done':>6} {'run':>5} {'queue':>6} {'fail':>5} {'cpu':>3} "
          f"{'submitted':11} {'runtime':8} hosts / name")
    for j in job_rows(c, a.all):
        n = counts(c, j["id"])
        spec = json.loads(j["spec"])
        hs = [r[0] for r in c.execute(f"select distinct host from shards where job=? and state in {ACTIVE_SHARD}", (j["id"],))]
        print(f"{j['id']:>4} {j['state']:9} {j['ntasks']:>6} {n.get('done', 0):>6} {n.get('assigned', 0):>5} "
              f"{n.get('queued', 0):>6} {n.get('failed', 0):>5} {spec['cpus']:>3} {fmt_t(j['submitted']):11} "
              f"{dur(j['submitted'], j['ended']):8} {','.join(hs) or '-'}  {j['name']}"
              + (f"  [bad: {','.join(json.loads(j['bad_hosts']))}]" if j["bad_hosts"] not in (None, "[]") else ""))


def ranges(spec):
    """'0-7,16' -> {0..7, 16}"""
    out = set()
    for part in spec.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out |= set(range(int(a), int(b or a) + 1))
    return out


def probe_topology(h):
    """Static CPU layout. Node hosts: lscpu (cores = distinct core/socket pairs) plus Intel hybrid P/E sets;
    Slurm hosts: sockets / cores / threads per node as Slurm counts them."""
    if h["backend"] == "node":
        rc, out, _ = sh(h, "LC_ALL=C lscpu -p=cpu,core,socket | grep -v '^#'; echo '#P'; cat /sys/devices/cpu_core/cpus "
                           "2>/dev/null; echo '#E'; cat /sys/devices/cpu_atom/cpus 2>/dev/null; true", timeout=30)
        if rc:
            return None
        body, _, rest = out.partition("#P")
        pset, _, eset = rest.partition("#E")
        rows = [l.split(",") for l in body.split()]
        core_of = {int(r[0]): (r[2], r[1]) for r in rows}
        t = {"cores": len(set(core_of.values())), "threads": len(core_of)}
        if eset.strip():  # hybrid: count physical cores in each set
            t["p"] = len({core_of[i] for i in ranges(pset) if i in core_of})
            t["e"] = len({core_of[i] for i in ranges(eset) if i in core_of})
        return t
    rc, out, _ = sh(h, f"sinfo -N -h -p {shlex.quote(h['partition'])} -O 'NodeList:40,Sockets:6,Cores:6,Threads:6' | sort -u",
                    timeout=30)
    if rc:
        return None
    nodes = {}
    for l in out.splitlines():
        f = l.split()
        if len(f) == 4 and all(x.isdigit() for x in f[1:]):
            s_, c_, t_ = map(int, f[1:])
            nodes[f[0]] = {"cores": s_ * c_, "threads": s_ * c_ * t_}
    return {"cores": sum(x["cores"] for x in nodes.values()), "threads": sum(x["threads"] for x in nodes.values()),
            "nodes": nodes}


def topology(hosts, max_age=7 * 86400):
    """Cached per host in STATE/topology.json (re-probed weekly or when missing)."""
    f = STATE / "topology.json"
    try:
        cache = json.loads(f.read_text())
    except (OSError, ValueError):
        cache = {}
    stale = [n for n, h in hosts.items() if h["enabled"] and time.time() - cache.get(n, {}).get("t", 0) > max_age]
    if stale:
        with cf.ThreadPoolExecutor(8) as ex:
            for n, t in zip(stale, ex.map(lambda n: probe_topology(hosts[n]), stale)):
                if t:
                    cache[n] = {**t, "t": time.time()}
        f.write_text(json.dumps(cache, indent=1))
    return cache


def topo_str(t):
    """24C24T (8P+16E) | 216C432T: 2×24C48T + 6×28C56T"""
    if not t:
        return ""
    s = f"{t['cores']}C{t['threads']}T"
    if "p" in t:
        s += f" ({t['p']}P+{t['e']}E)"
    kinds = {}
    for x in t.get("nodes", {}).values():
        k = f"{x['cores']}C{x['threads']}T"
        kinds[k] = kinds.get(k, 0) + 1
    if len(kinds) > 1:
        s += ": " + " + ".join(f"{n}×{k}" for k, n in sorted(kinds.items(), key=lambda x: int(x[0].split("C")[0])))
    elif kinds:
        s += f": {len(t['nodes'])}×{next(iter(kinds))}"
    return s


def host_lines(hosts, views, topo=None):
    out = []
    topo = topo or {}
    for n, h in sorted(hosts.items(), key=lambda x: x[1]["pref"]):
        v = views.get(n)
        if not h["enabled"]:
            out.append(f"{n:9} disabled")
            continue
        if not v or not v.get("ok"):
            out.append(f"{n:9} unreachable: {v.get('err', '') if v else ''}")
            continue
        q = " QUIET" if quiet_now(h) else ""
        if h["backend"] == "node":
            free, _ = slots({"cpus": 1, "mem_gb": 0}, h, v, {})
            out.append(f"{n:9} node   load {v['load']:5.1f}/{v['ncpu']:.0f}  RAM avail {v['mem_avail_gb']:5.0f}G  "
                       f"ours {v['ours_cpus']:>3}/{h['max_cpus']} cpus  free now {free} cpus{q}  [{topo_str(topo.get(n))}]")
        else:
            up = {k: r for k, r in v["nodes"].items() if r["up"]}
            idle = sum(r["cpu_idle"] for r in up.values())
            tot = sum(r["cpu_total"] for r in v["nodes"].values())
            mem = sum(r["mem_unalloc_gb"] for r in up.values())
            free, _ = slots({"cpus": 1, "mem_gb": 0}, h, v, {})
            out.append(f"{n:9} slurm  idle {idle:>4}/{tot} cpus on {len(up)}/{len(v['nodes'])} nodes  RAM unalloc {mem:5.0f}G  "
                       f"ours {v['ours_cpus']:>3}/{h['max_cpus']} cpus ({v['ours_pending']} pending)  free now {free} cpus{q}"
                       f"  [{topo_str(topo.get(n))}]")
            tn = (topo.get(n) or {}).get("nodes", {})
            for node, r in sorted(v["nodes"].items()):
                x = tn.get(node)
                g = f"GPU {r['gpus'] - r['gpus_used']}/{r['gpus']} free" if r.get("gpus") else ""
                out.append(f"  {node:12} {r['state']:7} {(str(x['cores']) + 'C' + str(x['threads']) + 'T') if x else '':9} "
                           f"idle {r['cpu_idle']:>3}/{r['cpu_total']:<3} RAM unalloc {r['mem_unalloc_gb']:5.0f}G  {g:16}"
                           f"ours {v.get('ours_node', {}).get(node, 0):>3}/{node_cap(h, r):<3} "
                           f"room {node_room(h, v, node) if r['up'] else 0}")
    return out


def cmd_status(a):
    _, hosts = load_config()
    c = db()
    views = gather(c, hosts)
    topo = topology(hosts)
    print("\n".join(host_lines(hosts, views, topo)))
    up = [n for n, h in hosts.items() if h["enabled"] and (views.get(n) or {}).get("ok")]
    ours = sum(views[n]["ours_cpus"] for n in up)
    cap = sum(hosts[n]["max_cpus"] for n in up)
    free = sum(slots({"cpus": 1, "mem_gb": 0}, hosts[n], views[n], {})[0] for n in up)
    tc = sum(topo.get(n, {}).get("cores", 0) for n in up)
    tt = sum(topo.get(n, {}).get("threads", 0) for n in up)
    print(f"{'total':9} {len(up)}/{sum(h['enabled'] for h in hosts.values())} hosts up  "
          f"ours {ours:>3}/{cap} cpus  free now {free} cpus  [{tc}C{tt}T]")


def cmd_plan(a):
    out, _ = locked(lambda: tick(dry=True))
    if not out:
        print("nothing to start")
    for j, host, idxs, lanes, node in out:
        print(f"job {j['id']} -> {host}{':' + node if node else ''}: {len(idxs)} tasks in {lanes} lanes "
              f"({idxs[0]}..{idxs[-1]})")


def get(c, jid):
    j = c.execute("select * from jobs where id=?", (jid,)).fetchone()
    if not j:
        sys.exit(f"no job {jid}")
    return j


def cmd_show(a):
    c = db()
    j = get(c, a.id)
    print(json.dumps({**dict(j), "spec": json.loads(j["spec"]), "tasks": counts(c, a.id)}, indent=1))
    for s in c.execute("select * from shards where job=? order by id", (a.id,)):
        print(f"shard {s['id']:>4} {s['host']:8} {s['state']:9} {len(json.loads(s['tasks'])):>5} tasks {s['lanes']:>3} lanes "
              f"{fmt_t(s['started'])} {dur(s['started'], s['ended']):8} {s['handle'] or ''} {s['note'] or ''}")


def cmd_tasks(a):
    c = db()
    get(c, a.id)
    q = "select * from tasks where job=?" + (" and state=?" if a.state else "") + " order by idx"
    for t in c.execute(q, (a.id, a.state) if a.state else (a.id,)):
        print(f"{t['idx']:>6} {t['state']:9} rc {'' if t['rc'] is None else t['rc']:>3} {t['host'] or '-':8} "
              f"shard {t['shard'] or '-':>4} tries {t['attempts']} {dur(t['started'], t['ended'])}")


def cmd_logs(a):
    c = db()
    get(c, a.id)
    t = c.execute("select * from tasks where job=? and idx=?", (a.id, a.task)).fetchone()
    if not t or not t["shard"]:
        sys.exit("task not started")
    s = c.execute("select * from shards where id=?", (t["shard"],)).fetchone()
    local = WORK / str(a.id) / str(s["id"]) / "log" / f"{a.task}.log"
    if s["state"] in ("done", "cancelled", "failed") and local.exists():
        print("".join(local.read_text().splitlines(keepends=True)[-a.n:]), end="")
        return
    _, hosts = load_config()
    rc, out, err = sh(hosts[s["host"]], f"tail -n {a.n} {shlex.quote(s['jobdir'])}/log/{a.task}.log")
    print(out or err, end="")


def cmd_cancel(a):
    def go():
        _, hosts = load_config()
        c = db()
        for jid in a.ids:
            get(c, jid)
            c.execute("update tasks set state='cancelled', ended=? where job=? and state='queued'", (time.time(), jid))
            for s in c.execute(f"select * from shards where job=? and state in {ACTIVE_SHARD}", (jid,)).fetchall():
                h = hosts[s["host"]]
                if s["state"] == "running":
                    if h["backend"] == "slurm":
                        sh(h, f"scancel {shlex.quote(s['handle'])}")
                    else:
                        sh(h, f"kill -TERM -- -$(cat {shlex.quote(s['jobdir'])}/pid); kill -CONT -- -$(cat {shlex.quote(s['jobdir'])}/pid)")
                    # leave the shard running; tick() sees the group gone / squeue empty, then fetches and settles
                    upd(c, "shards", s["id"], note="cancelling")
                elif s["state"] == "staging":
                    c.execute("update tasks set state='cancelled', ended=? where shard=? and state='assigned'", (time.time(), s["id"]))
                    upd(c, "shards", s["id"], state="cancelled", ended=time.time())
                else:
                    upd(c, "shards", s["id"], note="cancelling")
            c.commit()
            finish_job(c, jid)
            print(f"job {jid}: cancel sent")
    locked(go, "waiting for a running tick...")


def cmd_retry(a):
    def go():
        c = db()
        for jid in a.ids:
            get(c, jid)
            n = c.execute("update tasks set state='queued', rc=null, shard=null, host=null, ended=null, attempts=0 "
                          "where job=? and state in ('failed','cancelled')", (jid,)).rowcount
            if n:
                c.execute("update jobs set state='running', ended=null, bad_hosts='[]' where id=?", (jid,))
            c.commit()
            print(f"job {jid}: {n} tasks requeued")
    locked(go, "waiting for a running tick...")


def cmd_tick(a):
    locked(lambda: tick(), "waiting for a running tick...")


def cmd_init(a):
    if CONFIG.exists() and not a.force:
        sys.exit(f"{CONFIG} exists (--force overwrites it)")
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text((Path(__file__).parent / "csched.example.toml").read_text())
    print(f"wrote {CONFIG}: edit the [hosts.*] blocks, then try `csched status` and `ctop`")


def cmd_daemon(a):
    d, _ = load_config()
    log(f"csched daemon: poll every {d['poll_seconds']}s, config {CONFIG}")
    while True:
        t0 = time.time()
        try:
            locked(lambda: tick())
        except Exception as e:  # noqa: BLE001
            import traceback
            log(f"tick error: {e}\n{traceback.format_exc()}")
        _specs.clear()
        time.sleep(max(5, load_config()[0]["poll_seconds"] - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser(prog="csched", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"csched {__version__}")
    sp = ap.add_subparsers(dest="sub", required=True)
    s = sp.add_parser("init", help="write an example config")
    s.add_argument("--force", action="store_true")
    s = sp.add_parser("submit", help="queue a job")
    s.add_argument("-n", "--name")
    s.add_argument("-C", "--workdir", default=".", help="run here (synced to remote hosts at the same path under their home)")
    s.add_argument("-c", "--cpus", type=int, default=1, help="cores per task (OMP/MKL/OPENBLAS_NUM_THREADS set to this)")
    s.add_argument("--mem", type=float, default=0, help="GiB RAM per task (default mem_per_cpu_gb × cpus)")
    s.add_argument("-N", "--tasks", type=int, default=1, help="run the command this many times, $CSCHED_TASK = 0..N-1")
    s.add_argument("--task-file", help="one task per line; the line is $CSCHED_ARG ({arg} in the command)")
    s.add_argument("--hosts", help="comma list of allowed hosts (default: all enabled)")
    s.add_argument("--batch", type=int, help="tasks per lane (default auto, ≤ max_batch); 1 for long tasks")
    s.add_argument("--max-cpus", type=int, help="cap this job's cores in use at once")
    s.add_argument("-p", "--priority", type=int, default=0)
    s.add_argument("--retries", type=int, help="requeue a task whose rc never appeared (default from csched.toml)")
    s.add_argument("--time", help="Slurm time limit per lane (a lane runs ~batch tasks), e.g. 12:00:00")
    s.add_argument("--local-disk", dest="local_disk", help="Slurm node-local scratch per lane, POOL:GiB[,..] or none "
                   "(host default `local_disk`; mounted at /scratch/<pool> → $CSCHED_LOCAL, TMPDIR)")
    s.add_argument("--slurm-exclude", dest="slurm_exclude")
    s.add_argument("-e", "--env", action="append", metavar="K=V")
    s.add_argument("--no-sync", action="store_true", help="workdir already exists on every allowed host")
    s.add_argument("--sync-path", action="append", help="extra local path to copy to each host once per job")
    s.add_argument("--exclude", action="append", help="rsync exclude pattern for syncing")
    s.add_argument("-o", "--output", action="append", help="path relative to workdir copied back after each shard")
    s.add_argument("--setup-on", action="append", metavar="HOST=CMD", help="extra setup line on one host")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s = sp.add_parser("ls")
    s.add_argument("-a", "--all", action="store_true")
    s = sp.add_parser("status")
    s.add_argument("--nodes", action="store_true", help=argparse.SUPPRESS)  # per-node lines are always shown now
    sp.add_parser("plan")
    for name in ("show", "tasks"):
        s = sp.add_parser(name)
        s.add_argument("id", type=int)
        if name == "tasks":
            s.add_argument("--state")
    s = sp.add_parser("logs")
    s.add_argument("id", type=int)
    s.add_argument("task", type=int, nargs="?", default=0)
    s.add_argument("-n", type=int, default=50)
    for name in ("cancel", "retry"):
        s = sp.add_parser(name)
        s.add_argument("ids", type=int, nargs="+")
    sp.add_parser("tick")
    sp.add_parser("daemon")
    a = ap.parse_args()
    globals()["cmd_" + a.sub](a)


if __name__ == "__main__":
    main()
