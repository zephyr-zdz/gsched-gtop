---
name: gsched
description: Run GPU jobs through the user's gsched queue instead of picking a card by hand, and check GPU state with gtop. Use for any GPU job (training, evaluation, inference, sweeps), when the user asks which GPUs are free, or mentions gsched, gtop, ctop, GPU scheduling or queueing jobs on their servers or Slurm cluster.
---

# gsched + gtop

`gsched` is the user's GPU job queue: a controller (usually the `gsched` systemd user unit) polls the hosts in
`~/.config/gsched/hosts.toml` over ssh, starts queued jobs on idle GPUs, syncs the workdir over and copies logs
and outputs back. It manages only the user's own jobs and never touches other people's processes. `gtop` shows the
same hosts live.

## Look before you submit

```bash
gtop --once -w 160        # one plain-text snapshot: every GPU, its processes, which gsched job owns them, the queue
gsched status             # per host: each GPU idle / settling / gsched <ids> / mine (not gsched) / foreign <user>
gsched ls                 # active jobs + those that ended in the last 24 h (-a: all)
gsched plan               # what the next pass would start where (dry run)
```
Don't run the interactive `gtop` (curses) from an agent; use `--once` or `--json`.

## Submit

```bash
gsched submit -n NAME [opts] -- python train.py --cfg a.yaml      # prints the job id
gsched submit -n NAME -- "python prep.py && python train.py"      # one quoted string for shell syntax
gsched submit --file jobs.jsonl                                    # one {"cmd": ..., "name": ..., ...} per line
```
- `-C DIR` workdir (default: cwd). On other hosts the job runs at the same path under that host's home; the workdir
  is rsynced there first (`--exclude PAT` for big dirs such as `runs/`, `.venv/`, `--no-sync` if it already exists).
- `-g N` whole GPUs (default 1). `--gpu-mem GB` instead makes a 1-GPU job packable next to other `--gpu-mem` jobs;
  declare it honestly, nothing enforces it.
- `--cpus N`, `--mem GB` (host RAM), `--hosts a,b`, `--gpu-type X`, `-p N` priority, `--retries N` (only for jobs that
  resume from checkpoints), `-e K=V`, `--setup-on HOST=CMD` (e.g. activate an environment on one host).
- `-o PATH` (repeatable): path relative to the workdir copied back when the job ends. Write big outputs outside `~`
  if the user's server rules say so.
- Slurm hosts: `--time 2-00:00:00`, `--slurm-exclude NODE`, `--queue-ahead`.
- `--queue-ahead` [queue_ahead]: on a Slurm host with `queue_ahead = N` in hosts.toml, when no card is free the job
  is sbatched anyway and waits PENDING in Slurm (any allowed card type), so it is in line for the next freed card
  instead of racing other users' pending jobs. All our PENDING jobs per host are capped at `max_pending`
  (default: `queue_ahead`, else 2), whatever the reason: don't squat on cards.
- `--local-disk POOL:GiB[,POOL:GiB]` [local_disk]: Slurm node-local scratch. There is no default pool unless the host
  sets `local_disk`; `none` turns it off. The pool is mounted at `/scratch/<pool>` = `$GSCHED_LOCAL` (also
  `$GSCHED_LOCAL_<POOL>`); `TMPDIR`, `PYSCF_TMPDIR`, `GAUSS_SCRDIR`, `QCSCRATCH` and `PSI_SCRATCH` point to
  `$GSCHED_LOCAL/tmp` unless `-e` sets them. **Private to the job and deleted when it ends**: copy what you need
  into the workdir (and `-o` it back). Placement only picks a node with enough free space in each pool
  (`gsched status` shows "disk free"). Use it for I/O-heavy work, many small files, or QM scratch.
- In the command, `{home}`, `{workdir}` and `{host}` are replaced per host. The job gets `GSCHED_JOB`, `GSCHED_HOST`,
  `GSCHED_JOBDIR` and `CUDA_VISIBLE_DEVICES`.
- An agent that submits on its own behalf can use `--client NAME`; `[clients.NAME] max_gpus = {host = n}` in
  hosts.toml caps it.

## Follow up

```bash
gsched logs ID [-f] [-n 200]    # the job's out.log (remote while running, local copy afterwards)
gsched show ID                  # full record incl. spec, host, GPUs, note
gsched cancel ID... | gsched retry ID...
```
After submitting, check `gsched ls` / `gsched logs ID` and tell the user where each job landed.

## Slurm etiquette

One Slurm job per gsched job, named after it (`sbatch -J` = the gsched job's name, sanitized; `--comment` =
`gsched #<id> <client>`; there are no holder jobs). At most `max_pending` (default: `queue_ahead`, else 2) gsched
jobs sit PENDING in Slurm per host, whatever the reason. A host may set `max_time`; `launch` clamps `--time` to it,
and `submit` warns when `--time` exceeds it on every allowed Slurm host (node hosts have no limit). Give `--time`
close to the real runtime, or leave it out: gsched estimates it from past jobs with the same name stem (1.5 × the
max of the last 20, rounded to 30 min, 1 h – `max_time`; else the host's `default_time`) and warns when `--time` is
over 3 × the estimate.

## Rules

- Ask the user before queuing work that would hold several GPUs for more than about a day.
- Synced workdirs are copies: edits made on the remote side are lost unless listed with `-o`.
- A job without `--gpu-type` or `--hosts` may land on any enabled host, including a Slurm cluster whose software
  differs: pin `--hosts` when the command needs a local environment, or give the other hosts a `--setup-on`.
