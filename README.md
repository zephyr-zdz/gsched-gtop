# gsched / gtop / csched / ctop

[中文说明](README.zh-CN.md)

gsched is a personal or small-team GPU job queue for a few servers reached over SSH, with optional Slurm overflow. A controller polls hosts, debounces idle GPUs, observes other users' processes without touching them, and places jobs by tier with fallback and balancing. Small jobs that declare `--gpu-mem` can share a card. It rsyncs workdirs to hosts, tracks execution, fetches logs and declared outputs, and can requeue jobs lost after a reboot with `--retries`. For Slurm, it checks for a node with enough GPUs, CPUs, and RAM before calling sbatch, or optionally queues ahead. gtop is an nvtop-like live curses monitor for these hosts and the cards held by your running Slurm jobs, using `srun --overlap`; it associates processes with gsched jobs where identifiable. Both programs use only Python's standard library. No gsched installation is needed remotely: probes use python3 and nvidia-smi, alongside the Linux tools used for execution and transfers.

`csched` is the CPU counterpart: one command run for N tasks, cut into shards by free cores and RAM on node hosts and Slurm clusters, with the same ssh + rsync workflow. `ctop` is gtop's CPU page (the `ctop` entry point, `gtop --cpu`, or Tab in gtop).

- **No cluster software needed**: one controller, plain ssh + rsync to each GPU server, Python stdlib only, nothing
  installed on the servers.
- **Polite**: a GPU counts as idle only after it has been quiet for a while; other people's processes (and your own
  manual runs) are respected and never touched; optional hook for an external reservation board.
- **Tiers and fallback**: prefer your own boxes, spill over to shared ones or a Slurm cluster after a wait.
- **Packing**: small jobs that declare `--gpu-mem` share a card, best fit first.
- **Hands-off data movement**: the workdir is rsynced over, logs and `-o` outputs come back.
- **Quiet hours**, per-client GPU caps, priorities, retries after a reboot, JSONL batch submit.
- **gtop**: one screen for every GPU on every host, which process belongs to which job, plus the cards your Slurm jobs
  hold, and the queue.
- **csched / ctop**: a separate CPU queue (tasks, task files, node-local scratch) and its live CPU page.

```text
 gtop 14:02:11  every 2.0s
━ box1       load 9.8/24  RAM 61/125G  swap 0.0G  probe 0.2s
 GPU0 util ███████████████████▊  99%  mem ██████████████████▋· 22.4/24G  71°C 431/450W gsched #41 sweep-lr3e-4
        PID USER           SM GPUMEM ELAPSED    CPU    RSS JOB / COMMAND
    1840221 alice         97%  22.1G   3h12m   101%   9.7G #41 sweep-lr3e-4         python train.py --lr 3e-4
 GPU1 util ▏··················   1%  mem ····················  0.3/24G  34°C  21/450W idle
━ box2       load 2.1/64  RAM 40/256G  swap 0.0G  probe 0.6s
 GPU0 util ██████████████▌·····  73%  mem ████████████████····  38.9/48G  66°C 241/300W foreign bob
 GPU1 util ████████▊···········  44%  mem ███████▌············  18.2/48G  58°C 160/300W gsched #42 eval-a,#43 eval-b
━ cluster    slurm, tier 2; my running GPU jobs 1, 2 GPUs; gsched jobs here: 1
  gn12       mix    CPU ██▌····· 20/64 load 18   RAM ███····· 190/512G  GPU free H100 1/4
  gn12  load 18.0/64  RAM 183/512G  slurm 88123 gsched-44  1:02:37
 GPU1 util ██████████████████▌·  93%  mem ██████████████······ 57.1/80G  64°C 512/700W H100  gsched #44 pretrain
 GPU2 util ██████████████████··  90%  mem ██████████████······ 56.8/80G  63°C 498/700W H100  gsched #44 pretrain

━ gsched queue  queued 1  running 4   daemon active
     id state     host      gpus       submitted   runtime   rc  name / note
     41 running   box1      [0]        10-28 10:49 3h12m         sweep-lr3e-4
     42 running   box2      [1]        10-28 13:30 31m07s        eval-a
     ...
```

## Install

Use Linux and Python 3.11 or newer on the controller. Choose one installation method:

```sh
uv tool install git+https://github.com/zephyr-zdz/gsched-gtop
pipx install git+https://github.com/zephyr-zdz/gsched-gtop
pip install --user git+https://github.com/zephyr-zdz/gsched-gtop
```

Make sure the installed `gsched`, `gtop`, `csched` and `ctop` commands are on your PATH.
Configure SSH key authentication so each remote host works without a password prompt. SSH aliases and options from your SSH configuration work; a host can also set `ssh_opts`.
The controller needs ssh and rsync. Remote nodes need python3, nvidia-smi,
rsync for transfers, and normal Linux utilities such as bash, setsid, nohup, ps, and tail.
Your job's Python environment, libraries, datasets, and executables must also be available.
Slurm login hosts need the Slurm commands, including sbatch, sinfo, squeue, scancel,
and srun; compute nodes need python3 and nvidia-smi for monitoring.

## Quick start

Create the example configuration, then edit its host blocks for your machines:

```sh
gsched init
${EDITOR:-vi} ~/.config/gsched/hosts.toml
gsched status
gtop

csched init
${EDITOR:-vi} ~/.config/gsched/csched.toml
csched status
ctop
```

The generated example includes a local node and a placeholder SSH server. Remove or replace hosts you do not use. Set `home` when your remote home differs,
and use a host's `setup` command to initialize the environment before each job.
A minimal remote host block looks like this:

```toml
[hosts.lab]
backend = "node"
ssh = "me@gpu-server"
home = "/home/me"
tier = 1
max_gpus = 2
gpu_types = ["RTX_4090"]
```

From your project directory, queue a job and inspect the next placement:

```sh
gsched submit -n experiment --cpus 4 --mem 16 --retries 1 -o runs -- python train.py
gsched plan
gsched ls
gsched tick
```

Submission prints the job ID. `plan` is a dry run: it probes capacity but does not
launch jobs or advance their lifecycle. Idle history is saved by actual ticks,
so a freshly initialized controller may need multiple ticks before a GPU qualifies.
Submission alone does not run the scheduler. One tick starts or advances work;
staging, execution checks, and fetching need subsequent ticks.
Once the job starts, substitute its ID here:

```sh
gsched logs 1 -f
gsched show 1
```

For continuous scheduling, run `gsched daemon` in a persistent session or use the
provided [systemd user unit](examples/gsched.service). From a repository checkout:

```sh
mkdir -p ~/.config/systemd/user
cp examples/gsched.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gsched
```

Check the unit's `ExecStart` if your command is installed somewhere other than
`~/.local/bin/gsched`. `loginctl enable-linger "$USER"` keeps the user service
running after logout, where permitted. The example uses `KillMode=process`
so restarting the controller does not kill local jobs it has started.
Alternatively, invoke `gsched tick` manually or from cron, using an absolute
executable path. The default daemon polling interval is 60 seconds. `csched` has
its own unit, [`examples/csched.service`](examples/csched.service); install and
enable it the same way (`systemctl --user enable --now csched`).

## Submit jobs

Place submit options before `--`; everything after it is the command.
A single command string is treated as shell code; multiple arguments are shell-quoted.
For shell expansion or pipelines on the remote host, pass one quoted string:

```sh
gsched submit -C ~/research --gpu-mem 8 --exclude .venv/ -o results \
  -- 'python train.py {home}/data results'
gsched submit --hosts lab --setup-on 'lab=source ~/venv/bin/activate' \
  -e OMP_NUM_THREADS=4 -- python evaluate.py
```

| Submit option | Meaning |
| --- | --- |
| `-n`, `--name NAME` | Display name; otherwise the first 40 characters of the command. |
| `-C`, `--workdir PATH` | Workdir to sync and run in; default is the current directory. |
| `-g`, `--gpus N` | Number of GPUs; default 1. |
| `--gpu-mem GIB` | Declared GPU memory; enables packing for single-GPU node jobs. |
| `--cpus N` | Requested CPUs; default 4. |
| `--mem GIB` | Requested host RAM; default 0. |
| `--hosts A,B` | Explicit allowed host names from the configuration. |
| `--fallback` | Also allow hosts marked `optin = true`. |
| `--gpu-type TYPE` | Require a type listed in a host's `gpu_types`. |
| `-p`, `--priority N` | Higher numbers schedule first; default 0. |
| `--retries N` | Requeues allowed after a launched job disappears without an exit code; default 0. |
| `--time LIMIT` | Slurm wall-time limit, for example `2-00:00:00`. |
| `--queue-ahead` | Allow pending Slurm submission when that host also enables queue-ahead. |
| `--slurm-exclude NODES` | Comma-separated Slurm node names to avoid. |
| `-e`, `--env K=V` | Export an environment variable; repeatable. |
| `--no-sync` | Skip the workdir upload; the workdir must already exist remotely. |
| `--sync-path PATH` | Upload an additional file or directory; repeatable. |
| `--exclude PATTERN` | Exclude an rsync upload pattern; repeatable. |
| `-o`, `--output PATH` | Copy an output relative to the workdir back after execution; repeatable. |
| `--setup-on HOST=CMD` | Add a setup command for a particular host; repeatable. |
| `--client NAME` | Client label for accounting and caps; defaults to your local OS user. |
| `--file PATH` | Read one JSON job specification per nonblank line. |

Client names accept letters, digits, underscores, dots, and hyphens, up to 32 characters. JSONL fields override the defaults supplied by submit options. Use internal spec
keys such as `gpu_mem_gb`, `mem_gb`, `sync_paths`, `outputs`, and `setup_on`,
with lists or objects where appropriate, rather than hyphenated CLI names:

```jsonl
{"name":"run-1","cmd":"python train.py 1","gpu_mem_gb":8,"outputs":["runs/1"]}
{"name":"run-2","cmd":"python train.py 2","gpu_mem_gb":8,"outputs":["runs/2"]}
```

```sh
gsched submit -C ~/research --cpus 4 --mem 16 --file jobs.jsonl
```

### Paths, environment, and transfers

Commands, host `setup`, and per-job `setup_on` strings expand `{home}` to the
host's configured home, `{workdir}` to the remapped workdir, and `{host}` to its name.
Host setup runs first, then per-host job setup, then the command in the workdir.
The job receives `GSCHED_JOB` (ID), `GSCHED_HOST` (configured name), and
`GSCHED_JOBDIR` (execution metadata directory). Node jobs also receive
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` with assigned card indices.
Slurm supplies GPU visibility through its allocation; gsched does not set it there.
Use `--env` for additional exports.

Paths beneath the controller's home are remapped beneath each host's `home`.
For example, `/home/alice/research` becomes `/users/alice/research` on a host
with `home = "/users/alice"`. Paths outside the home remain unchanged.
For CLI workdirs and extra sync paths, `homeform` also folds targets of symlinks
immediately inside your home back into their home-relative spelling: if
`~/data` points to `/mnt/big/alice`, that target can travel as `~/data` on each host.
Create corresponding symlinks or directories on the destination as needed.
JSONL path overrides are used as supplied; use absolute paths in the specs.

Uploads copy directory contents to the corresponding remote path, or a file
into its corresponding parent directory. They use rsync without deletion.
`--exclude` applies to workdir and extra-path uploads. `--no-sync` still permits
explicit `--sync-path` uploads, and does not disable fetching logs or outputs.
Hosts with `ssh = "local"` execute locally and do not need upload staging.
Only one job's upload staging is scheduled per host at a time.

After execution, gsched fetches the remote job directory, including `out.log`,
and each declared `--output` back to its corresponding local workdir path.
Use separate output paths for concurrent experiments: transfers can overwrite files.
An unsuccessful output fetch is recorded in the job note; the terminal state
still follows the command's exit code. Inspect notes as well as `done` or `failed`.

## How scheduling decides

Queued jobs are considered by descending priority, then submission ID. The controller probes hosts concurrently. Unreachable hosts receive no placements; connectivity loss alone does not immediately fail their running jobs. The daemon rereads the host configuration each pass.

### Idle GPUs and packing

A node GPU must have no gsched jobs, no blocking compute processes, and no
reservation held by someone else. By default, memory must be below 2000 MiB
and utilization below 10%, continuously for `idle_seconds = 90`.
The compute-process size threshold is `foreign_ignore_mib = 1024` MiB.
Any compute process at or above it blocks idle placement, including your own
unmanaged processes. Smaller foreign processes are ignored for blocking and
their reported memory is subtracted from the idle memory test.
Other users' processes are only observed, never signalled or terminated.

Packing applies only to single-GPU node jobs declaring `--gpu-mem`.
An occupied card is eligible when all its gsched jobs declare memory, it has no
blocking foreign users or external reservations, and it is below
`max_jobs_per_gpu` (default 4). The fit check uses both actual free memory and
summed declarations, accounting for placements already planned in this pass,
and keeps `pack_headroom_gb` (default 4 GiB) free.
A best fit on an occupied card is preferred to opening an idle card. When a `--gpu-mem` job opens an idle card,
only cards with at least that much free memory qualify (useful on hosts with mixed cards).
There is no GPU memory enforcement.

### Tiers, balancing, and limits

Lower `tier` numbers are preferred. Without explicit `--hosts`, higher tiers
become eligible after `fallback_after_min` (default 10 minutes since submission).
`optin = true` hosts require explicit selection or `--fallback`.
`--hosts` skips the tier waiting delay; `--fallback` adds opt-in hosts even to
an explicit host list. `enabled = false` prevents new placements.

Within a tier, node idle-card placement favors the lowest fraction of
`max_gpus` already occupied by gsched, then the host with more idle cards.
Packing candidates are ranked first by remaining memory after the fit.
A host's `max_gpus` limits occupied cards, so packed jobs can share that budget. Before node placement, available RAM minus the request and this pass's plans
must leave `host_mem_reserve_gb` (default 16 GiB). One-minute load plus requested
and planned CPUs must not exceed the host's CPU count.
These node checks guide placement; they do not enforce CPU or RAM limits.

Per-client caps can reserve a smaller budget for an automated submitter:

```toml
[clients.my-agent]
max_gpus = { lab = 1, cluster = 2 }
```

Submit with `--client my-agent`. On node hosts a client's usage is the number of
distinct cards its jobs hold (including this pass' plans), so packing onto a card
the client already uses costs nothing; on Slurm hosts the requested GPU counts add up. An absent client/host entry adds no
client-specific limit. Host limits still apply. Clients share the controller's
account and database; client labels are not separate authenticated users.

### Quiet hours and reservations

Set `quiet_hours = "01:00-08:00"` on a node to prevent new placements during
that interval in the controller's local time. Running gsched process groups
receive SIGSTOP, then SIGCONT after the window. Jobs retain GPU memory while
paused. Quiet windows may cross midnight. Slurm jobs are not paused this way.

An optional `[defaults] reservations_cmd` runs on the controller during scheduler
gathering and prints JSON with this contract:

```json
{"lab":{"0":["alice"],"1":["bob","carol"]}}
```

Keys are configured host names and GPU indices; values are lists of holder names.
Holders matching the controller's OS user are removed; other holders block idle placement and packing on that card. This does not preempt a running job. A failing command is logged and treated as no reservations, so this integration
fails open. gtop refreshes its reservation view at most once every 30 seconds.

### Slurm overflow and job lifecycle

See the commented Slurm block in [hosts.example.toml](src/gsched/hosts.example.toml).
Configure `backend = "slurm"`, the login `ssh`, `partition`, `tier`, and `max_gpus`.
`gpu_types` orders preferred types; `gres_types` maps names to Slurm GRES names.
Placement normally looks for a responding idle/mixed node that has the requested
GPUs, idle CPUs, and unallocated RAM together, respecting `--slurm-exclude`.
It then uses sbatch with the resource request; it does not pin the checked node,
so Slurm makes the final placement and the job may still wait pending.
The time limit defaults to host `default_time`, or `1-00:00:00`.
With both host `queue_ahead = N` and job `--queue-ahead`, gsched can submit when
no immediate fit exists. It limits pending gsched jobs on that host to N for
this path; Slurm chooses the node. Without an explicit GPU type, this path
requests generic GPUs. Slurm uses its own resource and queue policies.

The usual lifecycle is `queued → staging → running → fetching → done/failed`.
Jobs without uploads skip staging; local jobs can skip fetching. A Slurm job is
already `running` in gsched after sbatch accepts it, with a note if Slurm has it pending.
`cancelled` is a terminal state set by `gsched cancel ID...`: it signals the job's process group (or `scancel`s it)
and stops a running upload or fetch; if the host can't be reached the job is left as is, so you can retry.
A missing node process without an exit code can requeue after a 60-second grace
period; a vanished Slurm job has a 120-second grace period. `--retries N` permits
N additional launches. Normal nonzero command exits are failed, not auto-retried.
Make commands safe to rerun and restore their own checkpoints after a reboot.

`gsched ls` includes active jobs and jobs ended in the last day; `gsched ls -a`
includes all jobs. `gsched show ID` prints the stored specification and metadata.
`gsched logs ID -n 100` reads the last 100 log lines; `-f` follows the remote log.
`gsched retry ID...` requeues terminal jobs and resets their attempt counts.

## gtop

Run `gtop` for the live curses view, or use a snapshot:

```sh
gtop --once -w 160
gtop --json
gtop --hosts lab --once
gtop --interval 5
```

`--once` prints text; `-w` / `--width` controls its line width. `--json` emits
raw per-host probe state, including errors. `--hosts` selects comma-separated
hosts to poll; omitted Slurm hosts can still appear as unpolled queue summaries.
`-d` / `--interval` sets the node probe interval (default 2 seconds, minimum 1).
`-a` / `--all` includes small and graphics processes in a text snapshot.
`--cpu` starts on the csched CPU page (also with `--once` / `--json`); the
`ctop` entry point does the same.

| Key | Action |
| --- | --- |
| `q` | Quit, or close the help/log view. |
| `j` / `k`, down / up | Scroll a line. |
| PageDown / PageUp | Scroll a page. |
| `g` / `G` | Go to the top / bottom. |
| `+` / `-` | Increase / decrease the probe interval. |
| `a` | Toggle small and graphics processes. |
| `c` | Toggle full command lines for your processes. |
| `l` | Ask for a gsched job ID and display its log. |
| Tab | Switch between the GPU and CPU (csched) pages. |
| `C` / `V` | Go to the CPU / GPU page directly. |
| `h` / `?` | Show help. |

GPU rows show utilization, memory, temperature, power draw/limit, and live
utilization history (`hist`, newest sample on the right). `idle` means eligible;
`settling` means the idle debounce is still running. `gsched #ID` identifies
managed jobs; `mine (not gsched)` means your unmanaged processes; `foreign`
identifies blocking users; `reserved` identifies holders; `busy` covers other load.
Process rows include PID, user, SM utilization, GPU memory, elapsed time, CPU,
RSS, and job/command. Other users' processes show process names only.
Process-to-job attribution walks parent processes and recognizes the default
`/gsched/jobs/ID/wrap.sh` path pattern; custom job directory layouts can lose it.

For Slurm, gtop probes cards held by your running GPU jobs, including jobs
submitted outside gsched. One streaming `srun --overlap` step per job normally
lasts for the session, rather than a new step per refresh. A failed step may be
restarted. Steps appear in accounting and count toward MaxStepCount.
The Slurm probe interval is fixed at startup to max(5 seconds, requested interval). Node summaries show Slurm-allocated CPUs/RAM and free GPUs, not just process use. gtop reads gsched's SQLite database without writing it or taking the tick lock.


## csched / ctop (CPU queue)

`csched` is a separate queue for CPU cores and RAM. A job is one command run for
N tasks (`-N N`, or one task per line of `--task-file`); each task gets `-c`
cores and `--mem` GiB. Every pass it probes the enabled hosts, slices the queued
tasks into shards by free capacity, launches them, and fetches results back.

```sh
csched init                    # write ~/.config/gsched/csched.toml
csched submit -n sweep -C . -N 12 -c 2 -o results/ -- python sweep.py {i}
csched submit -n mols -C . --hosts cluster -c 4 --task-file mols.txt -- bash run.sh {arg}
csched status                  # free cores / RAM per host; per-node lines on Slurm
csched ls [-a]                 # active + last 24 h jobs
csched plan                    # what the next pass would start where
csched show ID | tasks ID [--state S] | logs ID [TASK] [-n N]
csched cancel ID... | retry ID... | tick | daemon
```

Task variables: `$CSCHED_ARG` (the task-file line, `{arg}` in the command),
`$CSCHED_TASK` (`{i}`), `$CSCHED_JOB`, `$CSCHED_SHARD`, `$CSCHED_HOST`,
`$CSCHED_JOBDIR`, `$CSCHED_CPUS`, `$CSCHED_NTASKS`. `{home}`, `{workdir}` and
`{host}` are replaced per host, and `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` are
set to the task's core count.

- Node hosts run a shard under `xargs -P <lanes>` in a setsid process group.
  Slurm hosts submit one `sbatch --array` per shard, one element per lane, pinned
  to one node (`--nodelist`) when the placement code picked one.
- `pref` orders hosts (lower first), but every allowed host is filled in the
  same pass. Node room is capped by `max_cpus`, `reserve_cpus` and current load;
  Slurm room per node is `node_frac` of its cores minus `cores_per_free_gpu` per
  idle GPU. `--max-cpus` caps a single job's concurrency.
- `-C DIR` is rsynced to each allowed host once per job; `-o PATH` is copied back
  after each shard. `--no-sync`, `--sync-path`, `--exclude` work as in gsched.
- A task whose rc never appears (node died, time limit) is requeued `--retries`
  times (default from `[defaults]`). A real nonzero exit is not retried
  automatically; use `csched retry ID`. A shard that fails completely with
  rc 97/126/127, or while the same job succeeds elsewhere, marks that host bad
  for the job and moves its tasks to the remaining hosts.
- Slurm `--local-disk POOL:GiB[,POOL:GiB]` (or a host's `local_disk`) mounts
  node-local scratch at `/scratch/<pool>` as `$CSCHED_LOCAL`; `TMPDIR` and common
  QM scratch variables point to `$CSCHED_LOCAL/tmp` unless set with `-e`. There
  is no default pool unless the host configures one. Scratch is private to the
  lane and deleted when it ends, so copy results out with `-o`.

A minimal `~/.config/gsched/csched.toml`:

```toml
[defaults]
poll_seconds = 30
mem_per_cpu_gb = 2
max_batch = 8
retries = 2

[hosts.node1]                 # the controller's own machine
backend = "node"
ssh = "local"
pref = 5
max_cpus = 6
reserve_cpus = 2
mem_reserve_gb = 8
home = "/home/me"

[hosts.node2]
backend = "node"
ssh = "me@node2.example.org"
pref = 6
max_cpus = 8
home = "/home/me"

[hosts.cluster]               # optional Slurm cluster
backend = "slurm"
ssh = "me@login.cluster.example.org"
pref = 1
max_cpus = 64
node_frac = 1.0
cores_per_free_gpu = 0
partition = "cpu"
home = "/home/me"
jobroot = "/home/me/.local/state/csched/jobs"   # shared with the compute nodes
default_time = "1-00:00:00"
# local_disk = "ssd0:20"
```

`ctop` (or Tab in gtop) shows per enabled csched host: our cores / `max_cpus`,
QUIET during quiet hours, a load bar and available RAM on node hosts, and one
line per Slurm node with CPU allocation, unallocated RAM and state. Below is the
csched job table (active plus the last 24 h), like `csched ls`. The csched probe
runs in a background thread only while the CPU page is visible.

## Files and environment

| Setting / path | Purpose |
| --- | --- |
| `GSCHED_CONFIG` | Config path; default `~/.config/gsched/hosts.toml`. |
| `GSCHED_STATE` | State directory; default `~/.local/state/gsched`. |
| `GSCHED_WORK` | Local job directories and logs; default `$GSCHED_STATE/jobs`. |
| `gsched.db` in state | SQLite jobs and host idle history, using WAL mode. |
| `tick.lock` in state | Serializes scheduling passes and cancellation. |
| Host `jobroot` | Remote job directories, one numeric ID per job; defaults to the remapped local work path. |
| `CSCHED_CONFIG` | csched config; default `~/.config/gsched/csched.toml`. |
| `CSCHED_STATE` | csched state directory; default `~/.local/state/csched`. |
| `CSCHED_WORK` | csched job directories and logs; default `$CSCHED_STATE/jobs`. |
| `csched.db`, `topology.json` | csched jobs/shards/tasks and the cached CPU topology. |
| `~/.ssh/gsched-%C` | SSH ControlMaster sockets; connections persist for five minutes. |

Config and state defaults honor `XDG_CONFIG_HOME` and `XDG_STATE_HOME`.
Local job directories can contain `out.log`, scripts, staging/fetch logs, and
transfer return codes. Remote directories contain the run scripts, PID, start
time, and exit code. Keep the SQLite state and job directories available while
jobs are active. Use one controller and one state directory for a queue.

## Using AI coding agents

`gtop --once -w 160`, `gtop --json`, `gtop --cpu --once`, `gsched ls`, `gsched status`,
`csched ls`, and `csched status` work without
an interactive terminal. An agent can inspect capacity, submit with
`--client my-agent`, and check IDs, states, notes, and logs. Configure its own
`[clients.my-agent]` cap to limit how many cards it can occupy.

## Claude Code skill

[`skills/gsched/SKILL.md`](skills/gsched/SKILL.md) teaches a Claude Code agent to run GPU work through gsched instead
of picking cards by hand; [`skills/csched/SKILL.md`](skills/csched/SKILL.md) does the same for CPU work. Copy the
folders to `~/.claude/skills/` (or a project's `.claude/skills/`) and adapt the rules section to your servers.

## Tests

```sh
python -m unittest discover -s tests -v
```

The tests cover parsing, placement, packing, client caps, quiet hours, reservations, csched config/placement/task
files/scripts, the ctop CPU page and CLI round trips; none of them needs a GPU or ssh.

## Limits and license

GPU packing has no memory enforcement: declare `--gpu-mem` honestly and leave
headroom. gsched manages only jobs it launches; it observes existing work and
never cancels other users' processes. It is a single-controller queue backed by
SQLite, for Linux hosts, with no authentication layer beyond your SSH access.
Shared workdirs and overlapping outputs need coordination between submitters.
If the controller dies in the instant between starting a job and recording it, the next pass can start it a second
time; jobs that resume from checkpoints are the safe kind.

Licensed under the MIT license.
