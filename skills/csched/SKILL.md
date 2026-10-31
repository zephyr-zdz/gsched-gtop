---
name: csched
description: Run CPU jobs through the user's csched queue instead of ssh-ing around and nohup-ing by hand, and check CPU capacity with ctop. Use for CPU jobs, sweeps and task files, when the user asks which hosts have free cores, or mentions csched, ctop, CPU scheduling or the CPU queue. Separate from gsched, which handles GPUs.
---

# csched

`csched` is the user's **CPU** job queue, separate from gsched (GPUs). Its
controller runs on one machine (usually the `csched` systemd user unit) and polls
the hosts in `~/.config/gsched/csched.toml` over ssh. Every 30 s it slices each
job's queued tasks into shards by free cores and RAM, launches them, and fetches
results back. It manages only the user's own jobs.

A job is one command run for N tasks. Submit, then check where it landed:

```bash
csched submit -n NAME [opts] -- CMD...   # prints the job id
csched ls                                # active jobs + last 24 h (-a: all)
csched status                            # free cores / RAM per host; per-node lines on Slurm
csched plan                              # what the next pass would start where (dry run)
csched show ID | tasks ID [--state S] | logs ID [TASK] [-n N]
csched cancel ID... | retry ID...        # retry requeues failed/cancelled tasks
csched tick                              # one pass now instead of waiting for the daemon
```

## Look before you submit

```bash
csched status             # per host: free cores / RAM, our use, QUIET, Slurm per-node room
csched ls                 # active jobs and what ended in the last 24 h
csched plan               # what the next pass would start where
ctop --once -w 160        # one plain-text snapshot of the CPU page
```

Do not run the interactive `ctop` / `gtop` (curses) from an agent; use `--once`
or `--json`. `ctop` is gtop's CPU page; Tab switches pages inside gtop.

## Submit options that matter

- `-N N` tasks, or `--task-file FILE` (one task per line). The line is
  `$CSCHED_ARG` (`{arg}` in the command); the index is `$CSCHED_TASK` (`{i}`).
  Also set: `$CSCHED_JOB`, `$CSCHED_SHARD`, `$CSCHED_HOST`, `$CSCHED_JOBDIR`,
  `$CSCHED_CPUS`, `$CSCHED_NTASKS`.
- `-c N` cores per task (sets `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`), `--mem GB`
  per task (default `mem_per_cpu_gb` × cpus, normally 2).
- `-C DIR` workdir (default: cwd). It is rsynced to each allowed host **once per
  job**; `-o PATH` (relative to the workdir) is copied back after **each shard**.
  `--no-sync` if the workdir is already there; `--sync-path` and `--exclude` tune
  the rsync. Results land in the local work dir; task logs live under
  `~/.local/state/csched/jobs/<job>/<shard>/log/<i>.log`.
- Environments differ per host: a command that needs a specific Python env must
  pin `--hosts`, or add `--setup-on HOST=CMD` (or the host's `setup` line).
- `--batch N` tasks per lane (auto, ≤ `max_batch`, default 8; use `--batch 1`
  for long tasks), `--max-cpus N` caps the job's concurrency, `-p N` priority,
  `-e K=V`, `--time 12:00:00` (Slurm lane limit), `--retries N`,
  `--slurm-exclude NODE`.
- Slurm `--local-disk POOL:GiB` (or the host's `local_disk`; `none` turns it
  off) mounts node-local scratch at `/scratch/<pool>` as `$CSCHED_LOCAL` and
  points `TMPDIR` and common QM scratch variables at `$CSCHED_LOCAL/tmp` unless
  set with `-e`. It is private to the lane and deleted when the lane ends, so
  copy results out with `-o`. There is no default pool unless the host configures
  one.

## Behaviour worth knowing

- Node hosts run a shard under `xargs -P <lanes>`; Slurm hosts submit one
  `sbatch --array` with one element per lane, each element running its tasks in
  turn. On Slurm, shards are pinned to one node when the scheduler picks one.
- `pref` orders hosts, but every allowed host is filled in the same pass. Node
  room is capped by `max_cpus`, `reserve_cpus` and current foreign load; Slurm
  room per node is `node_frac` of its cores minus `cores_per_free_gpu` per idle
  GPU. `csched status` shows the details.
- A task whose rc never appears (node died, lane hit the time limit) is requeued
  `--retries` times. A real failure (rc ≠ 0) is **not** retried automatically —
  use `csched retry ID`.
- A shard that fails completely with rc 97/126/127 (cd failed, command not
  found), or while the same job succeeds on other hosts, marks that host bad for
  the job; its tasks go to the remaining hosts (`csched show ID` lists them).

## Examples

```bash
# sweep over a task file on a Slurm cluster with a prebuilt environment
csched submit -n sweep -C . --hosts cluster -c 4 --task-file jobs.txt \
  --setup-on cluster='export PATH=~/env/bin:$PATH' -o out/ -- bash run.sh {arg}

# a job with node-local scratch and an activated conda env
csched submit -n pyscf -C . --hosts cluster -c 8 --mem 16 --batch 1 \
  --setup-on cluster='conda activate qm' --local-disk ssd0:200 \
  -o results/ -- python compute.py {i}

# anywhere with free cores, no special env
csched submit -C . -N 12 -c 2 -o results/ -- python sweep.py {i}
```

## Rules

- Monitoring: `csched status` / `csched plan`; live view with `ctop` (also Tab
  inside `gtop`).
- Synced workdirs are copies: edits made on the remote side are lost unless
  listed in `-o`.
- Keep big outputs outside the synced workdir where the user's server rules say
  so; declare enough `--mem` and `-c` honestly.
- Do not raise a host's `max_cpus` or re-enable a disabled host without the
  user's OK.
- After submitting, check `csched ls` / `csched logs ID` and report where each
  job landed.
