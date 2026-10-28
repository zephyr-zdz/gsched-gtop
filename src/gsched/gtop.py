"""gtop: live GPU monitor for gsched's hosts, nvtop-style, stdlib only.
Plus Slurm hosts: the nodes and cards your running jobs hold (per node: allocated CPUs / RAM, free GPUs).

Per GPU: utilization history, memory, temperature, power, and what gsched thinks of the card
(idle / settling / gsched job / mine / foreign user, decided by gsched's own node_view). Per process:
user, pid, SM %, GPU memory, elapsed, CPU % / RSS, and the gsched job it belongs to. Below: the gsched queue.
Read-only: it never writes gsched's DB and never takes its tick lock.
Slurm hosts: one ssh per gtop session runs a small monitor on the login node; for each of your running GPU jobs it
starts one `srun --overlap` step that streams the same per-GPU probe from inside the job (one step per job and gtop
session, not one per refresh: steps count toward MaxStepCount and show in sacct). Refresh there: max(5 s, interval).

  gtop                 interactive (q quit, j/k/PgUp/PgDn/g/G scroll, +/- interval, a all procs,
                       c full command lines, l log of a job, h help)
  gtop --once          print one snapshot as plain text and exit
  gtop --json          dump the raw per-host probe as JSON and exit
"""
import argparse, curses, json, locale, os, re, shlex, sqlite3, subprocess, sys, threading, time
from collections import deque

from . import core as G  # gsched's config, ssh and node_view, so card labels match the scheduler

SMI_FIELDS = "index,uuid,memory.used,memory.total,utilization.gpu,name,temperature.gpu,power.draw,power.limit,fan.speed,clocks.sm"

# Runs on each host (python3 -). Same keys as gsched's PROBE so gsched.node_view() accepts it, plus details.
AGENT = r'''
import json, os, pwd, re, subprocess, sys
def q(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""
me = pwd.getpwuid(os.getuid()).pw_name
out = {"user": me, "ncpu": os.cpu_count(), "load1": os.getloadavg()[0], "node": os.uname().nodename,
       "step_gpus": os.environ.get("SLURM_STEP_GPUS", "")}  # inside a Slurm step: the job's real card indices
mi = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in open("/proc/meminfo"))}
out["mem_total_gb"] = mi["MemTotal"] / 2**20
out["mem_avail_gb"] = mi["MemAvailable"] / 2**20
out["swap_used_gb"] = (mi["SwapTotal"] - mi["SwapFree"]) / 2**20
out["smi"] = q("nvidia-smi", "--query-gpu=" + sys.argv[1], "--format=csv,noheader,nounits")
def owner(pid):
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except Exception:
        return "?"
procs = []
for line in q("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits").splitlines():
    p = [x.strip() for x in line.split(",")]
    if len(p) == 3 and p[1].isdigit():
        procs.append({"uuid": p[0], "pid": int(p[1]), "mib": int(p[2]) if p[2].isdigit() else 0, "user": owner(p[1])})
out["procs"] = procs
# pmon: per-process SM% / mem% / fb, graphics processes too; columns differ between drivers -> use the header
pm, names = [], []
for line in q("nvidia-smi", "pmon", "-c", "1", "-s", "um").splitlines():
    if line.startswith("# gpu") or line.startswith("#gpu"):
        names = line.lstrip("#").split()
    elif line.strip() and not line.startswith("#") and names:
        f = line.split(None, len(names) - 1)
        if len(f) == len(names):
            pm.append(dict(zip(names, f)))
out["pmon"] = pm
pids = sorted({int(x["pid"]) for x in pm if x.get("pid", "").isdigit()} | {x["pid"] for x in procs})
info = {}
if pids:
    for line in q("ps", "-o", "pid=,user:32=,etimes=,rss=,pcpu=,args=", "-p", ",".join(map(str, pids))).splitlines():
        f = line.split(None, 5)
        if len(f) < 5:
            continue
        pid, user = int(f[0]), f[1]
        d = {"user": user, "etime": int(f[2]), "rss_kb": int(f[3]), "cpu": float(f[4])}
        if user == me:  # only our own command lines; other users' processes show their process name only
            d["args"] = f[5] if len(f) > 5 else ""
            p, job = pid, None
            for _ in range(40):  # gsched jobs run under .../gsched/jobs/<id>/wrap.sh
                try:
                    m = re.search(r"/gsched/jobs/(\d+)/wrap\.sh", open(f"/proc/{p}/cmdline").read().replace("\0", " "))
                    if m:
                        job = int(m.group(1))
                        break
                    p = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
                except Exception:
                    break
                if p <= 1:
                    break
            d["job"] = job
        info[pid] = d
out["ps"] = info
print(json.dumps(out))
'''

# Runs inside a Slurm step (python3 -c LOOP fields interval count agent): the probe above every `interval` s.
LOOP = r'''
import os, sys, time
fields, iv, n, src = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
code = compile(src, "gtop-agent", "exec")
sys.argv = ["gtop-agent", fields]
while True:
    exec(code, {"__name__": "__main__"})
    sys.stdout.flush()  # srun gone -> BrokenPipe -> step ends
    n -= 1
    if n == 0:
        break
    time.sleep(iv)
'''

# Runs on a Slurm login node (python3 - args): keeps one streaming srun step per running GPU job of ours and prints
# one JSON line per interval; ends (and stops its steps) when gtop goes away or after one pass with "once".
SLURM_MON = r'''
import json, os, pwd, subprocess, sys, threading, time
a = json.loads(sys.argv[1])
me = pwd.getpwuid(os.getuid()).pw_name
iv, once = a["interval"], a["once"]
jobs, lock = {}, threading.Lock()
def q(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""
def reader(jid, j):
    for line in j["p"].stdout:
        try:
            d = json.loads(line)
            d["t"] = time.time()
            with lock:
                j["probes"][d["node"]] = d
        except Exception:
            if line.strip():
                j["err"] = line.strip()[-200:]
def start(jid, row):
    p = subprocess.Popen(["srun", f"--jobid={jid}", "--overlap", "-N", row["nnodes"], "--ntasks-per-node=1", "--mem=0",
                          "--cpu-bind=none", "-J", "gtop", "python3", "-u", "-c", a["loop"], a["fields"], str(iv),
                          "1" if once else "0", a["agent"]],
                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    j = jobs.setdefault(jid, {"probes": {}, "err": ""})
    j.update(row=row, p=p, started=time.time())
    j["thread"] = threading.Thread(target=reader, args=(jid, j), daemon=True)
    j["thread"].start()
def stop(j):
    if j["p"].poll() is None:
        j["p"].terminate()
        try:
            j["p"].wait(3)
        except Exception:
            j["p"].kill()
            j["p"].wait()
def emit():
    with lock:
        out = [{**j["row"], "err": j["err"], "running": j["p"].poll() is None, "probes": dict(j["probes"])}
               for j in jobs.values()]
    sys.stdout.write(json.dumps({"user": me, "t": time.time(), "jobs": out, "nodes": nodes}) + "\n")
    sys.stdout.flush()
tsq, nodes = 0, ""
try:
    while True:
        if time.time() - tsq >= max(iv, 15) or once:
            tsq = time.time()
            rows = {}
            nodes = q("sinfo", "-N", "-h", "-p", a["partition"], "-O", a["sinfo_fmt"]) if a["partition"] else ""
            for line in q("squeue", "-u", me, "-h", "-t", "R", "-o", "%i|%j|%N|%D|%M|%b").splitlines():
                f = line.strip().split("|")
                if len(f) == 6 and "gpu" in f[5]:
                    rows[f[0]] = dict(zip(("id", "name", "nodes", "nnodes", "elapsed", "gres"), f))
            for jid in [x for x in jobs if x not in rows]:
                stop(jobs.pop(jid))
            for jid, row in rows.items():
                j = jobs.get(jid)
                if j is None or (not once and j["p"].poll() is not None and time.time() - j["started"] > 60):
                    start(jid, row)  # new job, or its step ended: restart at most once a minute
                else:
                    j["row"] = row
        if once:
            for j in jobs.values():
                j["thread"].join(60)
            emit()
            break
        emit()
        if os.getppid() == 1:
            break
        time.sleep(iv)
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    for j in jobs.values():
        stop(j)
'''


# ---------------------------------------------------------------- data

class Poller:
    """One thread per host: probe every `interval` s, keep the last good result and the last error."""

    def __init__(self, hosts, interval):
        self.hosts, self.interval = hosts, interval
        self.state = {n: {"p": None, "t": 0, "err": None, "dt": 0} for n in hosts}
        self.hist = {}  # (host, gpu) -> deque of util; Slurm hosts: (host, "node:real index")
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.slurm_iv = max(5.0, interval)  # fixed for the session (the remote monitor gets it at start)
        self.procs = []

    def slurm_arg(self, n, once):
        return shlex.quote(json.dumps({"interval": self.slurm_iv, "once": once, "fields": SMI_FIELDS,
                                       "loop": LOOP, "agent": AGENT, "partition": self.hosts[n].get("partition", ""),
                                       "sinfo_fmt": G.SINFO_FMT}))

    def slurm_store(self, n, line, t0):
        p = json.loads(line)
        p["view"] = G.slurm_view({"nodes": p.get("nodes", ""), "squeue": ""})["nodes"]  # per-node GPUs, CPUs, RAM
        for j in p["jobs"]:
            for node, d in j["probes"].items():
                d["gpuinfo"] = smi_rows(d["smi"])
        with self.lock:
            last = self.state[n].get("p") or {"jobs": []}
            seen = {(j["id"], node): d["t"] for j in last["jobs"] for node, d in j["probes"].items()}
            self.state[n].update(p=p, t=time.time(), err=None, dt=time.time() - t0)
            for j in p["jobs"]:
                for node, d in j["probes"].items():
                    if seen.get((j["id"], node)) == d["t"]:
                        continue  # no new sample from this step since the last line
                    real = slurm_real_idx(d)
                    for f in d["gpuinfo"].values():
                        u = int(f[4]) if f[4].isdigit() else 0
                        self.hist.setdefault((n, f"{node}:{real.get(f[0], f[0])}"), deque(maxlen=400)).append(u)

    def slurm_stream(self, n):
        """Long-lived ssh to the Slurm login node; restarted with a pause if it drops."""
        h = self.hosts[n]
        while True:
            t0 = time.time()
            try:
                p = subprocess.Popen([*G.ssh_argv(h), h["ssh"], f"python3 - {self.slurm_arg(n, False)}"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.procs = [x for x in self.procs if x.poll() is None] + [p]
                p.stdin.write(SLURM_MON)
                p.stdin.close()
                other = ""  # last line that wasn't a monitor sample (stderr is merged into stdout)
                for line in p.stdout:
                    try:
                        self.slurm_store(n, line, t0)
                    except (ValueError, KeyError, TypeError, IndexError):
                        if line.strip() and not line.startswith("**"):
                            other = line.strip()
                        continue
                    t0 = time.time()
                    self.wake.set()
                p.wait()
                err = other[-200:] or f"monitor exited rc={p.returncode}"
            except OSError as e:
                err = str(e)
            with self.lock:
                self.state[n].update(err=err, terr=time.time())
            self.wake.set()
            time.sleep(15)

    def probe(self, n):
        h = self.hosts[n]
        if h["backend"] == "slurm":
            t0 = time.time()
            rc, out, err = G.sh(h, f"python3 - {self.slurm_arg(n, True)}", stdin=SLURM_MON, timeout=90)
            try:
                if rc:
                    raise RuntimeError((err or out).strip().splitlines()[-1][-200:] if (err or out).strip() else f"rc={rc}")
                self.slurm_store(n, out.strip().splitlines()[-1], t0)
            except Exception as e:
                with self.lock:
                    self.state[n].update(err=str(e) or type(e).__name__, terr=time.time())
            return
        t0 = time.time()
        rc, out, err = G.sh(h, f"python3 - {SMI_FIELDS}", stdin=AGENT, timeout=15)
        try:
            if rc:
                raise RuntimeError((err or out).strip().splitlines()[-1][-200:] if (err or out).strip() else f"rc={rc}")
            p = json.loads(out.strip().splitlines()[-1])
            p["gpuinfo"] = smi_rows(p["smi"])
            p["gpus"] = "\n".join(", ".join(f[:5]) for f in p["gpuinfo"].values())  # what gsched.node_view parses
            p["jobs"] = {}
        except Exception as e:
            with self.lock:
                self.state[n].update(err=str(e) or type(e).__name__, terr=time.time())
            return
        with self.lock:
            self.state[n].update(p=p, t=time.time(), err=None, dt=time.time() - t0)
            for f in p["gpuinfo"].values():
                u = int(f[4]) if f[4].isdigit() else 0
                self.hist.setdefault((n, f[0]), deque(maxlen=400)).append(u)

    def loop(self, n):
        while True:
            t0 = time.time()
            self.probe(n)
            self.wake.set()
            time.sleep(max(0.2, self.interval - (time.time() - t0)))

    def start(self):
        for n, h in self.hosts.items():
            threading.Thread(target=self.slurm_stream if h["backend"] == "slurm" else self.loop, args=(n,),
                             daemon=True).start()

    def stop(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()

    def once(self):
        ts = [threading.Thread(target=self.probe, args=(n,)) for n in self.hosts]
        for t in ts:
            t.start()
        for t in ts:
            t.join()


def smi_rows(text):
    """nvidia-smi --query-gpu=SMI_FIELDS output -> {index: fields}; malformed lines (error texts) are skipped."""
    n = SMI_FIELDS.count(",") + 1
    rows = ([x.strip() for x in line.split(",")] for line in text.strip().splitlines())
    return {f[0]: f for f in rows if len(f) == n and f[0].isdigit()}


def ro_db():
    path = G.STATE / "gsched.db"
    if not path.exists():
        return None
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    c.row_factory = sqlite3.Row
    return c


def read_db():
    """(active jobs, recently ended jobs, idle-since per host) from gsched's DB, read-only."""
    try:
        c = ro_db()
        if c is None:
            return [], [], {}
        try:
            act = c.execute(f"select * from jobs where state in {G.ACTIVE} order by id").fetchall()
            ended = c.execute("select * from jobs where state not in "
                              f"{G.ACTIVE} and ended > ? order by ended desc limit 8", (time.time() - 86400,)).fetchall()
            idle = {r["host"]: json.loads(r["data"]) for r in c.execute("select host, data from hoststate")}
            return act, ended, idle
        finally:
            c.close()
    except sqlite3.Error:
        return [], [], {}


_daemon = {"t": 0, "v": "?"}
_reserved = {"t": 0, "v": None}


def reservations(d, hosts):
    """reservations_cmd's answer per host (see gsched), refreshed every 30 s."""
    if time.time() - _reserved["t"] > 30 or _reserved["v"] is None:
        _reserved["v"], _reserved["t"] = G.reservations(d, hosts, quiet=True), time.time()
    return _reserved["v"]


def daemon_state():
    if time.time() - _daemon["t"] > 30:
        try:
            r = subprocess.run(["systemctl", "--user", "is-active", "gsched"], capture_output=True, text=True, timeout=5)
            _daemon["v"] = r.stdout.strip() or "?"
        except Exception:
            _daemon["v"] = "?"
        _daemon["t"] = time.time()
    return _daemon["v"]


# ---------------------------------------------------------------- formatting

SPARK = "▁▂▃▄▅▆▇█"
EIGHTHS = " ▏▎▍▌▋▊▉"


def bar(frac, w):
    frac = min(max(frac, 0.0), 1.0)
    n = frac * w
    full = int(n)
    part = int((n - full) * 8) if full < w else 0
    s = "█" * full + (EIGHTHS[part] if part else "")
    return s + "·" * (w - len(s))


def spark(vals, w):
    """(placeholder for samples not taken yet, sparkline of the last w samples)."""
    vals = list(vals)[-w:] if w > 0 else []
    return "·" * (w - len(vals)), "".join(SPARK[min(7, int(v / 100 * 8))] if v > 0 else " " for v in vals)


def gib(mib):
    return f"{mib / 1024:.1f}G"


def hms(s):
    s = int(s)
    if s >= 86400:
        return f"{s // 86400}d{s % 86400 // 3600:02d}h"
    if s >= 3600:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s"


def ago(t):
    return hms(time.time() - t) if t else "never"


def short_name(name):
    name = re.sub(r"^NVIDIA ", "", name)
    name = re.sub(r" (Blackwell|Workstation|Server|Edition|Max-Q)\b", "", name)
    return name.strip()


def util_color(u):
    return "green" if u < 50 else "yellow" if u < 90 else "red"


def num(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


class Opts:
    history = False  # util sparkline; only meaningful in the live view
    all_procs = False
    full_cmd = False


def slurm_real_idx(d):
    """Inside a Slurm step nvidia-smi counts the job's cards from 0; SLURM_STEP_GPUS has their real indices."""
    real = sorted((x for x in d.get("step_gpus", "").split(",") if x.strip().isdigit()), key=int)
    return {str(k): r for k, r in enumerate(real)} if len(real) == len(d.get("gpuinfo", {})) else {}


def gpu_block(L, name, f, util, used, total, lab, st, p, smi_i, hvals, W, opts, jobs_by_id, hdr):
    """One GPU line plus its process rows; returns whether the process header has been printed."""
    temp, pw, pl = f[6], num(f[7]), num(f[8])
    # fixed columns: name 6, util bar + % , mem bar + numbers, temp/power 18; the rest is util history
    bw = max(8, min(20, (W - 60) // 4))
    # util history: fixed width (does not grow), neutral colour, so it is not mistaken for a bar
    sw = min(40, max(0, W - 6 - (bw + 10) - (bw + 17) - 18 - 35)) if opts.history else 0
    pad, sp = spark(hvals, sw)
    u = util
    mfrac = used / total if total else 0
    L.append([(f" {name} ", "bold"),
              ("util ", ""), (bar(u / 100, bw), util_color(u)), (f"{u:4d}%", util_color(u)),
              ("  mem ", ""), (bar(mfrac, bw), util_color(mfrac * 100)),
              (f"{used / 1024:5.1f}/{total / 1024:.0f}G", ""),
              (f" {temp:>3}°C {pw:3.0f}/{pl:3.0f}W ", "red" if num(temp) >= 85 else ""),
              *([("hist ", "dim"), (pad, "dim"), (sp + " ", "")] if sw >= 10 else []),
              (lab, st)])
    pm = {}
    for x in p["pmon"]:
        pm.setdefault((x.get("gpu"), x.get("pid")), x)
    # processes on this GPU
    rows = []
    seen = set()
    for x in p["procs"]:
        if x["uuid"] == f[1] and (opts.all_procs or x["mib"] >= 64):
            rows.append((x["pid"], x["mib"], pm.get((smi_i, str(x["pid"])), {})))
            seen.add(x["pid"])
    for (gi, pid), x in pm.items():
        if gi == smi_i and pid.isdigit() and int(pid) not in seen:
            fb = int(x.get("fb", "0")) if x.get("fb", "").isdigit() else 0
            if opts.all_procs or ("C" in x.get("type", "") and fb >= 64):
                rows.append((int(pid), fb, x))
    rows.sort(key=lambda r: -r[1])
    if rows and not hdr:
        hdr = True
        L.append([(f"   {'PID':>8} {'USER':<12} {'SM':>4} {'GPUMEM':>6} {'ELAPSED':>7} {'CPU':>6} {'RSS':>6} JOB / COMMAND", "dim")])
    for pid, mib, x in rows:
        ps = p["ps"].get(str(pid), {})
        user = ps.get("user", "?")
        me = user == p["user"]
        job = ps.get("job")
        jtxt = ""
        if job:
            jr = jobs_by_id.get(job)
            jtxt = f"#{job} {jname(jr)}" if jr else f"#{job}"
        cmd = ps.get("args") or x.get("command", "?").strip()
        if not opts.full_cmd and me:
            cmd = re.sub(r"^\S*/(?=[^/\s]+(\s|$))", "", cmd)  # basename of the executable
        sm = x.get("sm", "-") + "%" if x.get("sm", "-").isdigit() else "-"
        line = (f"   {pid:>8} {user[:12]:<12} {sm:>4} {gib(mib):>6} {hms(ps['etime']) if 'etime' in ps else '-':>7} "
                f"{ps.get('cpu', 0):5.0f}% {ps.get('rss_kb', 0) / 2**20:5.1f}G ")
        L.append([(line, "dim" if not me else ""), (f"{jtxt[:24]:<24} " if jtxt else "", "cyan"),
                  (cmd, "" if me else "dim")])
    return hdr


def expand_nodes(s):
    """Slurm nodelist -> names: "gn[01-03,05],x1" -> gn01, gn02, gn03, gn05, x1."""
    out = []
    for pre, rng, plain in re.findall(r"([^,\[]+)\[([^\]]+)\]|([^,\[\]]+)", s or ""):
        if plain:
            out.append(plain)
            continue
        for part in rng.split(","):
            a, _, b = part.partition("-")
            out += [f"{pre}{i:0{len(a)}d}" for i in range(int(a), int(b or a) + 1)] if a.isdigit() else [pre + part]
    return out


def nodes_block(L, h, nodes, W):
    """One line per Slurm node: free GPUs by type, allocated CPUs (load) and RAM, CPU model from hosts.toml."""
    labels, cpu_models = h.get("gpu_labels", {}), h.get("cpu_models", {})
    for node, r in sorted(nodes.items()):
        gp = ", ".join(f"{labels.get(f'{node}:{t}', t)} {r['free'].get(t, 0)}/{k}" for t, k in sorted(r["gres"].items()))
        if not r["cpu_total"]:
            L.append([(f"  {node:<10}", "bold"), (f" {r['state']:<6}", "dim"), (f" GPU free {gp}", "")])
            continue
        cf = r["cpu_alloc"] / r["cpu_total"]
        mf = 1 - r["mem_unalloc_gb"] / r["mem_total_gb"] if r["mem_total_gb"] else 0
        ld = f" load {r['load']:.0f}" if r.get("load") is not None else ""
        L.append([(f"  {node:<10}", "bold"), (f" {r['state']:<6}", "" if r["up"] else "red"),
                  (f" CPU {bar(cf, 8)} {r['cpu_alloc']:>2}/{r['cpu_total']:<2}{ld:<8}", util_color(cf * 100)),
                  (f" RAM {bar(mf, 8)} " + f"{r['mem_total_gb'] - r['mem_unalloc_gb']:.0f}/{r['mem_total_gb']:.0f}G".rjust(9),
                   util_color(mf * 100)),
                  (f"  GPU free {gp or '-'}", "" if r["up"] and any(r["free"].values()) else "dim"),
                  (f"  {cpu_models[node]}" if node in cpu_models else "", "dim")])


def slurm_block(L, n, h, s, act, jobs_by_id, hist, now, poller, W, opts):
    """A Slurm host: a one-line summary, plus the cards of our running GPU jobs when there are any."""
    p = s["p"]
    mj = [j for j in act if j["host"] == n]
    tag = f"slurm, tier {h['tier']}{', opt-in' if h['optin'] else ''}"
    head = [("━ ", "dim"), (f"{n:<9}", "bold")]
    if not p:
        L.append(head + [(f"  {tag}; " + (f"UNREACHABLE: {s['err']}" if s["err"] else "probing…"),
                          "red" if s["err"] else "dim"), (f"  gsched jobs here: {len(mj)}", "dim")])
        return
    ngpu = sum(len(d["gpuinfo"]) for j in p["jobs"] for d in j["probes"].values())
    head += [(f"  {tag}; my running GPU jobs {len(p['jobs'])}" + (f", {ngpu} GPUs" if ngpu else "")
              + f"; gsched jobs here: {len(mj)}", "" if p["jobs"] else "dim")]
    if s["err"]:
        head += [(f"  UNREACHABLE: {s['err']}", "red"), (f"  (data from {ago(s['t'])} ago)", "dim")]
    elif now - s["t"] > 3 * poller.slurm_iv + 10:
        head += [(f"  last update {ago(s['t'])} ago", "yellow")]
    L.append(head)
    mine = {x for j in p["jobs"] for x in [*expand_nodes(j["nodes"]), *j["probes"]]}
    nodes_block(L, h, {k: r for k, r in (p.get("view") or {}).items() if k in mine}, W)
    if not p["jobs"]:
        return
    first = len(L)
    gs = {j["handle"]: j for j in jobs_by_id.values() if j["host"] == n and j["handle"]}
    labels = h.get("gpu_labels", {})
    hdr = False
    for j in sorted(p["jobs"], key=lambda j: int(j["id"]) if j["id"].isdigit() else 0):
        gj = gs.get(j["id"])
        lab, st = (f"gsched #{gj['id']} {jname(gj)}", "cyan") if gj else (f"mine (not gsched): {j['name']}", "yellow")
        gtype = (j["gres"].split("gpu:")[1].split(":")[0].split(",")[0] if "gpu:" in j["gres"] else "")
        if not j["probes"]:
            L.append([(f"  {j['nodes']}", "bold"), (f"  slurm {j['id']} {j['name']}  {j['elapsed']}  ", "dim"), (lab, st),
                      (f"  {j['err'] or ('starting monitor step…' if j['running'] else 'monitor step ended')}",
                       "red" if j["err"] else "dim")])
            continue
        for node, d in sorted(j["probes"].items()):
            late = now - d["t"] > 3 * poller.slurm_iv + 10
            L.append([(f"  {node}", "bold"), (f"  load {d['load1']:.1f}/{d['ncpu']}", ""),
                      (f"  RAM {d['mem_total_gb'] - d['mem_avail_gb']:.0f}/{d['mem_total_gb']:.0f}G", ""),
                      (f"  slurm {j['id']} {j['name']}  {j['elapsed']}", "dim"),
                      (f"  no data for {ago(d['t'])}" + (f": {j['err']}" if j["err"] else ""), "yellow") if late else ("", "")])
            sub = len(L)
            real = slurm_real_idx(d)
            for i, f in sorted(d["gpuinfo"].items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
                r = real.get(i, f"?{i}")
                card = labels.get(f"{node}:{gtype}") or short_name(f[5])
                hdr = gpu_block(L, f"GPU{r}", f, int(num(f[4])), num(f[2]), num(f[3]), f"{card}  {lab}", st, d, i,
                                hist.get((n, f"{node}:{r}"), []), W, opts, jobs_by_id, hdr)
            if late:
                L[sub:] = [[(t, "dim") for t, _ in seg] for seg in L[sub:]]
    if s["err"]:
        L[first:] = [[(t, "dim") for t, _ in seg] for seg in L[first:]]


def build(poller, cfg, W, opts):
    """Lines of (text, style) segments for one frame; width W."""
    d, hosts = cfg
    act, ended, idle = read_db()
    jobs_by_id = {j["id"]: j for j in [*act, *ended]}
    L = []
    now = time.time()
    with poller.lock:
        state = {n: dict(s) for n, s in poller.state.items()}
        hist = {k: list(v) for k, v in poller.hist.items()}
    nodes = {n: h for n, h in hosts.items() if h["backend"] == "node"}

    for n, h in nodes.items():
        s = state[n]
        p = s["p"]
        stale = s["err"] is not None or not p
        head = [("━ ", "dim"), (f"{n:<9}", "bold")]
        if p:
            head += [(f"  load {p['load1']:.1f}/{p['ncpu']}", ""),
                     (f"  RAM {p['mem_total_gb'] - p['mem_avail_gb']:.0f}/{p['mem_total_gb']:.0f}G", ""),
                     (f"  swap {p['swap_used_gb']:.1f}G", "")]
        if s["err"]:
            head += [(f"  UNREACHABLE: {s['err']}", "red"), (f"  (data from {ago(s['t'])} ago)" if p else "", "dim")]
        elif p:
            late = opts.history and now - s["t"] > 2 * poller.interval + 5  # --once waits for Slurm hosts: not late
            head += [(f"  probe {s['dt']:.1f}s" + (f", {ago(s['t'])} ago" if late else ""), "yellow" if late else "dim")]
        else:
            head += [("  probing…", "dim")]
        L.append(head)
        first = len(L)
        if not p:
            L.append([])
            continue
        mine = [j for j in act if j["host"] == n and j["state"] in ("staging", "running")]
        views = G.node_view(h, p, d, idle.get(n, {}), now, mine, reservations(d, nodes).get(n, {}))
        hdr = False
        for g in views:
            i = str(g["index"])
            f = p["gpuinfo"].get(i, [""] * 11)
            if g["jobs"]:
                lab, st = "gsched " + ",".join(f"#{j['id']} {jname(j)}" for j in g["jobs"]), "cyan"
            elif g["foreign"]:
                lab, st = "foreign " + ",".join(g["foreign"]), "magenta"
            elif g["reserved"]:
                lab, st = "reserved " + ",".join(g["reserved"]), "magenta"
            elif g["mine_other"]:
                lab, st = "mine (not gsched)", "yellow"
            elif g["idle"]:
                lab, st = "idle", "green"
            elif g["idle_since"]:
                lab, st = f"settling {hms(now - g['idle_since'])}", "green"
            else:
                lab, st = "busy", "yellow"
            hdr = gpu_block(L, f"GPU{i}", f, g["util"], g["used"], g["total"], lab, st, p, i, hist.get((n, i), []),
                            W, opts, jobs_by_id, hdr)
        if stale:
            L[first:] = [[(t, "dim") for t, _ in seg] for seg in L[first:]]
        L.append([])

    # Slurm hosts: polled through the login node (see Poller.slurm_stream); left out with --hosts -> summary only
    for n, h in G.load_config()[1].items():
        if h["backend"] != "slurm":
            continue
        if n in hosts:
            slurm_block(L, n, h, state[n], act, jobs_by_id, hist, now, poller, W, opts)
        else:
            mj = [j for j in act if j["host"] == n]
            L.append([("━ ", "dim"), (f"{n:<9}", "bold"),
                      (f"  slurm, tier {h['tier']}{', opt-in' if h['optin'] else ''}; not polled (--hosts);"
                       f" gsched jobs here: {len(mj)}", "dim")])
    L.append([])
    # gsched queue
    counts = {}
    for j in act:
        counts[j["state"]] = counts.get(j["state"], 0) + 1
    dst = daemon_state()
    L.append([("━ gsched queue  ", "bold"),
              ("  ".join(f"{k} {v}" for k, v in counts.items()) or "empty", ""),
              (f"   daemon {dst}", "green" if dst == "active" else "red")])
    if act or ended:
        L.append([(f"   {'id':>4} {'state':9} {'host':9} {'gpus':10} {'submitted':11} {'runtime':8} {'rc':>3}  name / note", "dim")])
    sc = {"queued": "yellow", "staging": "cyan", "running": "green", "fetching": "cyan",
          "done": "dim", "failed": "red", "cancelled": "dim"}
    for j in [*act, *ended]:
        L.append([(f"   {j['id']:>4} ", ""), (f"{j['state']:9}", sc.get(j["state"], "")),
                  (f" {j['host'] or '-':9} {(j['gpus'] or '')[:10]:10} {G.fmt_t(j['submitted']):11} "
                   f"{G.dur(j['started'], j['ended']):8} {'' if j['rc'] is None else j['rc']:>3}  {jname(j)}", ""),
                  (f"  [{j['note']}]" if j["note"] else "", "dim")])
    return L


def jname(j):
    """Job name, prefixed with its client when it isn't ours (e.g. [my-agent])."""
    cl = G.client_of(j)
    return (f"[{cl}] " if cl != G.USER else "") + j["name"]


def job_log(jid, n=200):
    try:
        c = ro_db()
        j = c.execute("select * from jobs where id=?", (jid,)).fetchone() if c else None
    except sqlite3.Error:
        j = None
    if not j:
        return [f"no job {jid}"]
    local = G.WORK / str(jid) / "out.log"
    if j["state"] in ("done", "failed", "cancelled") and local.exists():
        return local.read_text(errors="replace").splitlines()[-n:]
    if not j["jobdir"]:
        return [f"job {jid} has not started ({j['state']})"]
    h = G.load_config()[1][j["host"]]
    rc, out, err = G.sh(h, f"tail -n {n} {shlex.quote(j['jobdir'])}/out.log", timeout=15)
    return (out or err).splitlines() or ["(empty)"]


HELP = """gtop keys
  q            quit (or close this / the log view)
  j k ↓ ↑      scroll        PgDn PgUp  page        g G  top / bottom
  + -          probe interval up / down (now {iv:.1f}s)
  a            also list small / graphics processes (now {a})
  c            full command lines for your processes (now {c})
  l            show the out.log of a gsched job (asks for the id)
  h ?          this help

GPU line: util bar, memory bar, temperature, power draw/limit, and gsched's
view of the card: idle / settling (idle for less than idle_seconds) / gsched #job / mine (not gsched) /
foreign <user> / reserved <holder> / busy. 'hist' = util history, one cell per probe, newest on the right
(· = not sampled yet). Process rows: pid, user, SM %, GPU memory, elapsed, CPU %, RSS, gsched job, command.
Other users' processes show only their process name. Hosts and thresholds come from gsched's hosts.toml.

Slurm hosts: only the cards your running GPU jobs hold, probed from inside each job by one
`srun --overlap` step per job for the life of this gtop (GPU<n> = the node's real card index);
refreshed every max(5 s, interval) at start, squeue every 15 s. For each node your jobs run on, one line from sinfo:
state (* = not responding), CPUs and RAM allocated by Slurm (load = 1-min load), free GPUs by type, CPU model."""


# ---------------------------------------------------------------- UI

STYLES = {}


def init_styles():
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    for k, (i, col) in {"green": (1, curses.COLOR_GREEN), "yellow": (2, curses.COLOR_YELLOW), "red": (3, curses.COLOR_RED),
                        "cyan": (4, curses.COLOR_CYAN), "magenta": (5, curses.COLOR_MAGENTA)}.items():
        curses.init_pair(i, col, bg)
        STYLES[k] = curses.color_pair(i)
    STYLES.update({"": curses.A_NORMAL, "bold": curses.A_BOLD, "dim": curses.A_DIM})


def draw(scr, lines, top, title):
    scr.erase()
    H, W = scr.getmaxyx()

    def put(y, segs):
        x = 0
        for text, st in segs:
            if x >= W - 1 or not text:
                continue
            try:
                scr.addnstr(y, x, text, W - 1 - x, STYLES.get(st, 0))
            except curses.error:
                pass
            x += len(text)

    put(0, title)
    for y, segs in enumerate(lines[top:top + H - 1], 1):
        put(y, segs)
    scr.noutrefresh()
    curses.doupdate()


def prompt(scr, text):
    H, W = scr.getmaxyx()
    curses.echo()
    scr.timeout(-1)
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    try:
        scr.move(H - 1, 0)
        scr.clrtoeol()
        scr.addnstr(H - 1, 0, text, W - 1, curses.A_REVERSE)
        s = scr.getstr(H - 1, len(text), 12).decode(errors="replace").strip()
    except curses.error:
        s = ""
    finally:
        curses.noecho()
        try:
            curses.curs_set(0)
        except curses.error:
            pass
    return s


def ui(scr, poller, cfg):
    init_styles()
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    scr.timeout(200)
    opts = Opts()
    opts.history = True
    top, mode, logjob, loglines, logt = 0, "main", None, [], 0
    last = 0
    while True:
        H, W = scr.getmaxyx()
        now = time.time()
        if now - last >= 0.5 or poller.wake.is_set():
            poller.wake.clear()
            last = now
            if mode == "log" and now - logt > max(2, poller.interval):
                loglines, logt = job_log(logjob, 500), now
            if mode == "main":
                lines = build(poller, cfg, W, opts)
            elif mode == "log":
                lines = [[(l.expandtabs(), "")] for l in loglines]
            else:
                lines = [[(l, "")] for l in HELP.format(iv=poller.interval, a=opts.all_procs, c=opts.full_cmd).splitlines()]
            top = max(0, min(top, len(lines) - (H - 1)))
            sub = {"main": "", "log": f"  log of job {logjob}  (q back)", "help": "  help  (q back)"}[mode]
            title = [(" gtop ", "bold"), (time.strftime("%H:%M:%S"), ""), (f"  every {poller.interval:.1f}s", "dim"),
                     (sub, "cyan"), ("   q quit  h help  l log  a all  c cmd  +/- interval", "dim")]
            draw(scr, lines, top, title)
        k = scr.getch()
        if k == -1:
            continue
        last = 0  # redraw right away after a key
        page = max(1, H - 2)
        if k in (ord("q"), 27):
            if mode == "main":
                return
            mode, top = "main", 0
        elif k in (ord("j"), curses.KEY_DOWN):
            top += 1
        elif k in (ord("k"), curses.KEY_UP):
            top = max(0, top - 1)
        elif k in (curses.KEY_NPAGE, ord(" ")):
            top += page
        elif k == curses.KEY_PPAGE:
            top = max(0, top - page)
        elif k in (ord("g"), curses.KEY_HOME):
            top = 0
        elif k in (ord("G"), curses.KEY_END):
            top = 10 ** 6
        elif k in (ord("+"), ord("=")):
            poller.interval = min(60.0, poller.interval * 1.5)
        elif k == ord("-"):
            poller.interval = max(1.0, poller.interval / 1.5)
        elif k == ord("a"):
            opts.all_procs = not opts.all_procs
        elif k == ord("c"):
            opts.full_cmd = not opts.full_cmd
        elif k in (ord("h"), ord("?")):
            mode, top = "help", 0
        elif k == ord("l"):
            s = prompt(scr, "log of job id: ")
            scr.timeout(200)
            if s.isdigit():
                mode, logjob, logt = "log", int(s), 0
                loglines, top = ["loading…"], 10 ** 6
        elif k == curses.KEY_RESIZE:
            pass


def main():
    ap = argparse.ArgumentParser(prog="gtop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--interval", type=float, default=2.0, help="seconds between probes per host (default 2)")
    ap.add_argument("--hosts", help="comma list of hosts (default: every host in gsched's hosts.toml)")
    ap.add_argument("--once", action="store_true", help="print one snapshot as plain text and exit")
    ap.add_argument("--json", action="store_true", help="print the raw per-host probe as JSON and exit")
    ap.add_argument("-a", "--all", action="store_true", help="also list small / graphics processes")
    ap.add_argument("-w", "--width", type=int, help="line width for --once (default: terminal width)")
    a = ap.parse_args()
    d, hosts = G.load_config()
    hosts = {n: h for n, h in hosts.items() if h["backend"] in ("node", "slurm")}
    if a.hosts:
        want = a.hosts.split(",")
        bad = [n for n in want if n not in hosts]
        if bad:
            sys.exit(f"unknown host(s) {', '.join(bad)}; known: {', '.join(hosts)}")
        hosts = {n: hosts[n] for n in want}
    poller = Poller(hosts, max(1.0, a.interval))
    if a.json:
        poller.once()
        print(json.dumps({n: s for n, s in poller.state.items()}, indent=1, default=str))
        return
    if a.once:
        poller.once()
        opts = Opts()
        opts.all_procs = a.all
        W = a.width or (os.get_terminal_size().columns if sys.stdout.isatty() else 160)
        for segs in build(poller, (d, hosts), W, opts):
            print("".join(t for t, _ in segs)[:W].rstrip())
        return
    locale.setlocale(locale.LC_ALL, "")
    os.environ.setdefault("ESCDELAY", "25")
    poller.start()
    try:
        curses.wrapper(ui, poller, (d, hosts))
    except KeyboardInterrupt:
        pass
    finally:
        poller.stop()  # the Slurm monitor stops its srun steps when its ssh goes away


if __name__ == "__main__":
    main()
