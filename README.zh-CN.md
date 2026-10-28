# gsched / gtop

[English](README.md)

gsched 是面向个人或小团队的 GPU 作业队列，适用于通过 SSH 访问的少量服务器，也可将作业分流到 Slurm。控制器会轮询主机，确认 GPU 持续空闲一段时间后再分配作业；它只观察其他用户的进程，不做干预，并按层级调度，支持回退和负载均衡。声明了 `--gpu-mem` 的小作业可以共用一张卡。gsched 通过 rsync 将工作目录同步到主机，跟踪执行状态，取回日志和指定输出；设置 `--retries` 后，还能将因重启而丢失的作业重新入队。对于 Slurm，它会先检查是否有节点同时满足 GPU、CPU 和内存需求，再调用 sbatch，也可选择提前排队。gtop 是类似 nvtop 的实时 curses 监控工具，可查看这些主机，以及通过 `srun --overlap` 监控正在运行的 Slurm 作业所占用的显卡；在能够识别时，它会将进程关联到 gsched 作业。两个程序都只依赖 Python 标准库。远程主机无需安装 gsched：探测使用 python3 和 nvidia-smi，执行和传输则使用常见 Linux 工具。

- **无需集群软件**：一个控制器，通过普通 ssh + rsync 连接各台 GPU 服务器，只依赖 Python 标准库，服务器上无需
  安装 gsched。
- **不干扰他人**：GPU 持续空闲一段时间后才会被视为空闲；尊重其他人的进程（以及你自己
  手动启动的任务），绝不干预；可选接入外部预约系统。
- **分层与回退**：优先使用自己的机器，等待一段时间后再分流到共享机器或 Slurm 集群。
- **显卡共享**：声明了 `--gpu-mem` 的小作业可共用一张卡，优先选择最合适的卡。
- **自动传输数据**：通过 rsync 上传工作目录，取回日志和 `-o` 指定的输出。
- **静默时段**、按客户端限制 GPU 数量、优先级、重启后重试、JSONL 批量提交。
- **gtop**：一屏查看所有主机上的所有 GPU、进程与作业的对应关系，以及 Slurm 作业
  占用的显卡和队列。

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

## 安装

控制器需使用 Linux 和 Python 3.11 或更新版本。选择一种方式安装：

```sh
uv tool install git+https://github.com/zephyr-zdz/gsched-gtop
pipx install git+https://github.com/zephyr-zdz/gsched-gtop
pip install --user git+https://github.com/zephyr-zdz/gsched-gtop
```

确保安装后的 `gsched` 和 `gtop` 命令位于 PATH 中。
配置 SSH 密钥认证，确保连接各台远程主机时无需输入密码。支持 SSH 配置中的别名和选项；也可以为主机设置 `ssh_opts`。
控制器需要 ssh 和 rsync。远程节点需要 python3、nvidia-smi、
用于传输的 rsync，以及 bash、setsid、nohup、ps、tail 等常见 Linux 工具。
作业所需的 Python 环境、库、数据集和可执行文件也必须就绪。
Slurm 登录主机需要 sbatch、sinfo、squeue、scancel、
srun 等 Slurm 命令；计算节点需要 python3 和 nvidia-smi，以便监控。

## 快速开始

生成示例配置，然后根据自己的机器修改主机配置块：

```sh
gsched init
${EDITOR:-vi} ~/.config/gsched/hosts.toml
gsched status
gtop
```

生成的示例包含一个本地节点和一个占位 SSH 服务器。删除或替换不使用的主机。如果远程主目录不同，请设置 `home`，
并使用主机的 `setup` 命令在每个作业开始前初始化环境。
最简远程主机配置如下：

```toml
[hosts.lab]
backend = "node"
ssh = "me@gpu-server"
home = "/home/me"
tier = 1
max_gpus = 2
gpu_types = ["RTX_4090"]
```

在项目目录中提交作业，并查看下一次调度的分配计划：

```sh
gsched submit -n experiment --cpus 4 --mem 16 --retries 1 -o runs -- python train.py
gsched plan
gsched ls
gsched tick
```

提交后会输出作业 ID。`plan` 仅作预演：探测容量，但不会
启动作业或推进其生命周期。空闲历史由实际的调度轮次保存，
因此刚初始化的控制器可能需要执行多轮，GPU 才能满足空闲条件。
仅提交作业不会运行调度器。一次 tick 会启动或推进任务；
上传准备、执行检查和结果取回需要后续 tick 才能完成。
作业启动后，将下面的 ID 替换为实际 ID：

```sh
gsched logs 1 -f
gsched show 1
```

如需持续调度，可在持久会话中运行 `gsched daemon`，或使用
提供的 [systemd 用户服务单元](examples/gsched.service)。在仓库检出目录中执行：

```sh
mkdir -p ~/.config/systemd/user
cp examples/gsched.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gsched
```

如果命令安装在 `~/.local/bin/gsched` 以外的位置，
请检查服务单元中的 `ExecStart`。在系统允许的情况下，`loginctl enable-linger "$USER"` 可让用户服务
在退出登录后继续运行。示例使用 `KillMode=process`，
因此重启控制器不会终止它已启动的本地作业。
也可以手动执行 `gsched tick`，或通过 cron 调用；请使用可执行文件的
绝对路径。守护进程默认每 60 秒轮询一次。

## 提交作业

提交选项放在 `--` 前面，后面的所有内容都是要执行的命令。
单个命令字符串会作为 shell 代码处理；多个参数则分别进行 shell 引用。
如需在远程主机上使用 shell 展开或管道，请传入一个带引号的字符串：

```sh
gsched submit -C ~/research --gpu-mem 8 --exclude .venv/ -o results \
  -- 'python train.py {home}/data results'
gsched submit --hosts lab --setup-on 'lab=source ~/venv/bin/activate' \
  -e OMP_NUM_THREADS=4 -- python evaluate.py
```

| 提交选项 | 含义 |
| --- | --- |
| `-n`, `--name NAME` | 显示名称；默认使用命令的前 40 个字符。 |
| `-C`, `--workdir PATH` | 要同步并在其中运行的工作目录；默认为当前目录。 |
| `-g`, `--gpus N` | GPU 数量；默认为 1。 |
| `--gpu-mem GIB` | 声明所需显存；为节点上的单 GPU 作业启用显卡共享。 |
| `--cpus N` | 请求的 CPU 数量；默认为 4。 |
| `--mem GIB` | 请求的主机内存；默认为 0。 |
| `--hosts A,B` | 显式指定允许使用的主机名，名称须来自配置。 |
| `--fallback` | 同时允许使用标记为 `optin = true` 的主机。 |
| `--gpu-type TYPE` | 要求使用主机 `gpu_types` 中列出的类型。 |
| `-p`, `--priority N` | 数值越大越优先调度；默认为 0。 |
| `--retries N` | 已启动的作业消失且没有退出码时，允许重新入队的次数；默认为 0。 |
| `--time LIMIT` | Slurm 运行时限，例如 `2-00:00:00`。 |
| `--queue-ahead` | 主机也启用提前排队时，允许提交进入 Slurm 等待队列。 |
| `--slurm-exclude NODES` | 要避开的 Slurm 节点名，以逗号分隔。 |
| `-e`, `--env K=V` | 导出环境变量；可重复指定。 |
| `--no-sync` | 跳过工作目录上传；远程工作目录必须已存在。 |
| `--sync-path PATH` | 上传额外的文件或目录；可重复指定。 |
| `--exclude PATTERN` | 排除匹配该 rsync 模式的上传内容；可重复指定。 |
| `-o`, `--output PATH` | 执行结束后，将相对于工作目录的输出复制回来；可重复指定。 |
| `--setup-on HOST=CMD` | 为指定主机添加初始化命令；可重复指定。 |
| `--client NAME` | 用于用量统计和配额限制的客户端标签；默认为本地操作系统用户名。 |
| `--file PATH` | 从文件中读取 JSON 作业描述，每个非空行一条。 |

客户端名称允许使用字母、数字、下划线、点和连字符，最长 32 个字符。JSONL 字段会覆盖提交选项提供的默认值。请使用内部作业描述
键名，如 `gpu_mem_gb`、`mem_gb`、`sync_paths`、`outputs` 和 `setup_on`，
并按需使用列表或对象，不要使用带连字符的命令行选项名：

```jsonl
{"name":"run-1","cmd":"python train.py 1","gpu_mem_gb":8,"outputs":["runs/1"]}
{"name":"run-2","cmd":"python train.py 2","gpu_mem_gb":8,"outputs":["runs/2"]}
```

```sh
gsched submit -C ~/research --cpus 4 --mem 16 --file jobs.jsonl
```

### 路径、环境与传输

命令、主机 `setup` 和作业的 `setup_on` 字符串中，`{home}` 会展开为
主机配置的主目录，`{workdir}` 展开为映射后的工作目录，`{host}` 展开为主机名。
先执行主机初始化命令，再执行作业针对该主机的初始化命令，最后在工作目录中执行作业命令。
作业会获得 `GSCHED_JOB`（ID）、`GSCHED_HOST`（配置中的主机名）和
`GSCHED_JOBDIR`（执行元数据目录）。节点作业还会获得
`CUDA_DEVICE_ORDER=PCI_BUS_ID`，以及包含所分配显卡索引的 `CUDA_VISIBLE_DEVICES`。
Slurm 通过自身的资源分配设置 GPU 可见性，gsched 不会为它设置这些变量。
可通过 `--env` 导出其他环境变量。

控制器主目录下的路径会映射到各主机的 `home` 下。
例如，在配置了 `home = "/users/alice"` 的主机上，
`/home/alice/research` 会变成 `/users/alice/research`。主目录之外的路径保持不变。
对于命令行指定的工作目录和额外同步路径，`homeform` 还会将主目录
直属符号链接的目标路径还原为主目录下的路径：例如，
若 `~/data` 指向 `/mnt/big/alice`，该目标路径可在各主机上按 `~/data` 传输。
请按需在目标主机上创建对应的符号链接或目录。
JSONL 中覆盖的路径会按原样使用；请在作业描述中使用绝对路径。

上传目录时，会将目录内容复制到对应的远程路径；上传文件时，
会将文件复制到对应的父目录。传输使用 rsync，不删除目标端文件。
`--exclude` 对工作目录和额外路径的上传都生效。`--no-sync` 仍允许
显式指定的 `--sync-path` 上传，也不会禁用日志或输出取回。
配置了 `ssh = "local"` 的主机在本地执行，无需上传准备。
每台主机同一时间只安排一个作业进行上传准备。

执行结束后，gsched 会取回远程作业目录（包括 `out.log`），
并将每个 `--output` 指定的输出取回到本地工作目录中的对应路径。
并行实验请使用不同的输出路径：传输可能覆盖文件。
输出取回失败会记录在作业备注中；最终状态
仍由命令的退出码决定。除了查看 `done` 或 `failed`，也要检查备注。

## 调度决策方式

排队作业按优先级从高到低处理，同优先级按提交 ID 排序。控制器并发探测各主机。无法连接的主机不会被分配新作业；仅连接中断不会立即将其运行中的作业判为失败。守护进程每轮都会重新读取主机配置。

### 空闲 GPU 与显卡共享

节点上的 GPU 必须没有 gsched 作业、没有阻止调度的计算进程，
也没有被其他人预约。默认要求显存占用低于 2000 MiB、
利用率低于 10%，并持续满足 `idle_seconds = 90`。
计算进程的显存阈值为 `foreign_ignore_mib = 1024` MiB。
任何达到或超过该阈值的计算进程都会阻止空闲卡分配，包括你自己
未由 gsched 管理的进程。低于阈值的外部进程不会阻止调度，
且空闲显存检查会扣除它们报告的显存占用。
其他用户的进程仅被观察，绝不会收到信号或被终止。

显卡共享仅适用于声明了 `--gpu-mem` 的单 GPU 节点作业。
已占用的显卡须满足以下条件才能继续分配：卡上所有 gsched 作业均已声明显存，
没有会阻止调度的外部用户进程或外部预约，且作业数低于
`max_jobs_per_gpu`（默认为 4）。容量检查同时考虑实际空闲显存和
声明显存的总和，计入本轮已计划的分配，
并保留 `pack_headroom_gb`（默认为 4 GiB）空闲显存。
优先选择已占用显卡中最合适的一张，再考虑启用空闲卡。带 `--gpu-mem` 的作业启用空闲卡时，
只有空闲显存达到声明值的卡才符合条件（适用于显卡型号混用的主机）。
系统不会强制限制显存用量。

### 层级、负载均衡与限制

`tier` 数值越小越优先。未显式指定 `--hosts` 时，
提交后经过 `fallback_after_min`（默认为 10 分钟），才允许使用更高层级的主机。
`optin = true` 的主机需要显式选择，或使用 `--fallback`。
`--hosts` 会跳过层级等待时间；`--fallback` 会将需主动选择的主机加入候选范围，
即使已显式指定主机列表也一样。`enabled = false` 会阻止新作业分配。

同一层级内，为节点分配空闲卡时，优先选择 gsched 已占用卡数
占 `max_gpus` 比例最低的主机，再选择空闲卡更多的主机。
显卡共享候选首先按分配后的剩余显存排序。
主机的 `max_gpus` 限制的是已占用卡数，因此共享显卡的作业可共用这部分配额。向节点分配作业前，可用内存扣除本次请求和本轮计划用量后，
必须还剩 `host_mem_reserve_gb`（默认为 16 GiB）。一分钟平均负载加上本次请求
和本轮计划的 CPU 数量，不得超过主机的 CPU 总数。
这些节点检查用于指导调度，不会强制限制 CPU 或内存用量。

可通过客户端配额，为自动提交程序设置较小的资源额度：

```toml
[clients.my-agent]
max_gpus = { lab = 1, cluster = 2 }
```

提交时使用 `--client my-agent`。在节点主机上，客户端用量按其作业
占用的不同显卡数量计算（包括本轮计划），因此向该客户端已使用的卡
追加共享作业不会消耗额外配额；在 Slurm 主机上，则累加请求的 GPU 数量。未配置客户端或主机条目时，
不施加对应的客户端限制。主机限制仍然有效。各客户端共用控制器的
账号和数据库；客户端标签不是独立认证的用户。

### 静默时段与预约

为节点设置 `quiet_hours = "01:00-08:00"`，可在控制器本地时间的
该时段内停止分配新作业。运行中的 gsched 进程组
会收到 SIGSTOP，时段结束后收到 SIGCONT。作业暂停期间仍占用显存。
静默时段可以跨越午夜。Slurm 作业不会以这种方式暂停。

可选配置 `[defaults] reservations_cmd`，在调度器
收集信息时于控制器上执行，并按以下格式输出 JSON：

```json
{"lab":{"0":["alice"],"1":["bob","carol"]}}
```

键为配置中的主机名和 GPU 索引，值为预约者名称列表。
与控制器操作系统用户名相同的预约者会被移除；其他预约者会阻止该卡上的空闲分配和显卡共享。这不会抢占运行中的作业。命令失败时会记录日志，并按无预约处理，因此此集成
出错时仍允许调度。gtop 的预约信息最多每 30 秒刷新一次。

### 分流到 Slurm 与作业生命周期

参见 [hosts.example.toml](src/gsched/hosts.example.toml) 中带注释的 Slurm 配置块。
配置 `backend = "slurm"`、登录主机的 `ssh`、`partition`、`tier` 和 `max_gpus`。
`gpu_types` 按优先顺序列出类型；`gres_types` 将名称映射到 Slurm GRES 名称。
正常分配时，会寻找可响应、处于空闲或混合状态的节点，要求该节点同时满足请求的
GPU、空闲 CPU 和未分配内存需求，并遵守 `--slurm-exclude`。
随后使用 sbatch 提交资源请求；不会绑定到检查过的节点，
因此最终由 Slurm 决定分配位置，作业仍可能需要排队。
运行时限默认取主机的 `default_time`，未设置时为 `1-00:00:00`。
同时设置主机的 `queue_ahead = N` 和作业的 `--queue-ahead` 后，即使
当前没有合适资源，gsched 也可提交作业。通过这种方式提交时，该主机上等待中的 gsched 作业数
最多为 N；节点由 Slurm 选择。未显式指定 GPU 类型时，这种方式
会请求通用 GPU。Slurm 使用自身的资源和队列策略。

通常的生命周期为 `queued → staging → running → fetching → done/failed`。
无需上传的作业会跳过 staging；本地作业可以跳过 fetching。Slurm 作业在
sbatch 接受后，gsched 就将其标记为 `running`；若仍在 Slurm 中排队，会在备注中说明。
`cancelled` 是由 `gsched cancel ID...` 设置的终态：该命令会向作业进程组发送信号（或通过 `scancel` 取消作业），
并停止正在进行的上传或取回；如果无法连接主机，作业保持原样，便于重试。
节点进程消失且没有退出码时，可在 60 秒宽限期后重新入队；
Slurm 作业消失时的宽限期为 120 秒。`--retries N` 允许
额外启动 N 次。命令正常退出但退出码非零时，会判为失败，不会自动重试。
请确保命令可安全重复执行，并能在重启后自行从检查点恢复。

`gsched ls` 包含活跃作业和最近一天结束的作业；`gsched ls -a`
包含所有作业。`gsched show ID` 输出保存的作业描述和元数据。
`gsched logs ID -n 100` 读取最后 100 行日志；`-f` 持续跟踪远程日志。
`gsched retry ID...` 将已进入终态的作业重新入队，并重置尝试次数。

## gtop

运行 `gtop` 打开实时 curses 界面，也可查看快照：

```sh
gtop --once -w 160
gtop --json
gtop --hosts lab --once
gtop --interval 5
```

`--once` 输出文本；`-w` / `--width` 控制行宽。`--json` 输出
各主机的原始探测状态，包括错误。`--hosts` 选择要轮询的主机，
以逗号分隔；未选中的 Slurm 主机仍可能以未经轮询的队列摘要形式显示。
`-d` / `--interval` 设置节点探测间隔（默认 2 秒，最短 1 秒）。
`-a` / `--all` 在文本快照中包含小进程和图形进程。

| 按键 | 操作 |
| --- | --- |
| `q` | 退出，或关闭帮助/日志视图。 |
| `j` / `k`, down / up | 滚动一行。 |
| PageDown / PageUp | 滚动一页。 |
| `g` / `G` | 跳到顶部/底部。 |
| `+` / `-` | 增大/减小探测间隔。 |
| `a` | 切换是否显示小进程和图形进程。 |
| `c` | 切换是否显示自己进程的完整命令行。 |
| `l` | 输入 gsched 作业 ID 并显示日志。 |
| `h` / `?` | 显示帮助。 |

GPU 行显示利用率、显存、温度、功耗/功率上限，以及实时
利用率历史（`hist`，最新采样在右侧）。`idle` 表示符合分配条件；
`settling` 表示仍在等待空闲状态稳定。`gsched #ID` 表示
受管理的作业；`mine (not gsched)` 表示你未由 gsched 管理的进程；`foreign`
表示阻止调度的用户；`reserved` 表示预约者；`busy` 表示其他负载。
进程行包含 PID、用户、SM 利用率、显存、已运行时间、CPU、
RSS 和作业/命令。其他用户的进程只显示进程名。
进程与作业的关联通过追溯父进程实现，并识别默认的
`/gsched/jobs/ID/wrap.sh` 路径模式；自定义作业目录结构可能导致无法关联。

对于 Slurm，gtop 会探测你正在运行的 GPU 作业所占用的显卡，包括
未通过 gsched 提交的作业。每个作业通常只启动一个持续传输数据的 `srun --overlap` 步骤，
并在整个会话期间保持运行，而非每次刷新都新建步骤。步骤失败后可能会
重新启动。这些步骤会出现在作业统计中，并计入 MaxStepCount。
Slurm 探测间隔在启动时固定为 5 秒与请求间隔中的较大值。节点摘要显示 Slurm 已分配的 CPU/内存和空闲 GPU，而不只是进程用量。gtop 只读取 gsched 的 SQLite 数据库，不写入，也不获取 tick 锁。

## 文件与环境

| 设置 / 路径 | 用途 |
| --- | --- |
| `GSCHED_CONFIG` | 配置路径；默认为 `~/.config/gsched/hosts.toml`。 |
| `GSCHED_STATE` | 状态目录；默认为 `~/.local/state/gsched`。 |
| `GSCHED_WORK` | 本地作业目录和日志；默认为 `$GSCHED_STATE/jobs`。 |
| 状态目录中的 `gsched.db` | 存储作业和主机空闲历史的 SQLite 数据库，使用 WAL 模式。 |
| 状态目录中的 `tick.lock` | 保证各轮调度与取消操作串行执行。 |
| 主机的 `jobroot` | 远程作业目录，每个作业使用一个数字 ID 目录；默认为映射后的本地作业路径。 |
| `~/.ssh/gsched-%C` | SSH ControlMaster 套接字；连接保持五分钟。 |

配置和状态目录的默认值遵循 `XDG_CONFIG_HOME` 和 `XDG_STATE_HOME`。
本地作业目录可包含 `out.log`、脚本、上传准备/取回日志，以及
传输返回码。远程目录包含运行脚本、PID、开始
时间和退出码。作业活跃期间，请保持 SQLite 状态和作业目录可用。
每个队列只使用一个控制器和一个状态目录。

## 使用 AI 编程智能体

`gtop --once -w 160`、`gtop --json`、`gsched ls` 和 `gsched status` 无需
交互式终端即可运行。智能体可以查看容量，使用
`--client my-agent` 提交，并检查 ID、状态、备注和日志。为它单独配置
`[clients.my-agent]` 配额，即可限制其占用的显卡数量。

## Claude Code 技能

[`skills/gsched/SKILL.md`](skills/gsched/SKILL.md) 指导 Claude Code 智能体通过 gsched 运行 GPU 任务，
无需手动选择显卡。将该文件夹复制到 `~/.claude/skills/`（或项目的 `.claude/skills/`），并根据
服务器情况调整规则部分。

## 测试

```sh
python -m unittest discover -s tests -v
```

测试覆盖解析、分配、显卡共享、客户端配额、静默时段、预约，以及完整的命令行流程；所有测试
均无需 GPU 或 ssh。

## 限制与许可证

显卡共享不会强制限制显存：请如实声明 `--gpu-mem`，并预留
余量。gsched 只管理自己启动的作业；对已有任务仅作观察，
绝不取消其他用户的进程。它是面向 Linux 主机、基于
SQLite 的单控制器队列，除 SSH 访问权限外没有额外认证层。
使用共享工作目录或重叠输出路径时，提交者之间需要协调。
如果控制器恰好在启动作业后、记录作业前崩溃，下一轮可能再次启动同一作业；
能够从检查点恢复的作业更安全。

采用 MIT 许可证。
