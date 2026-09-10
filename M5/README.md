# M5 运行方法

## 完整小说文本

本仓库收录了 M5 单次正式实验生成的六部长篇小说：
[`texts/20260905_luna_medium`](texts/20260905_luna_medium)。每种方法均包含
`chapter_001.md` 至 `chapter_320.md`，合计 1,920 个章节文件。六组实验共享同一故事设定、
全局写作要求、320 章章节任务与 10 个预设长程钩子；正文作者均为
`gpt-5.6-luna`、`medium` 推理强度，上下文预算均为 12,000 tokens。各组仅改变
写作当前章时可获得的历史记忆形式，作者不可读取未来章节提示词或钩子金标。

| 目录 | 实验方法 | 写作当前章时提供的历史上下文 |
|---|---|---|
| [`B0`](texts/20260905_luna_medium/B0) | Current-only | 不提供历史章节或历史摘要；仅使用共同故事设定、全局要求和当前章任务。 |
| [`B1`](texts/20260905_luna_medium/B1) | Recency-window | 从最近一章开始逆序装入历史正文，直到用满 12,000-token 记忆预算；不能主动召回更早章节。 |
| [`B2`](texts/20260905_luna_medium/B2) | Running-summary | 每章结束后将既有摘要与新正文合并为一份滚动摘要；下一章只获得该累计摘要。 |
| [`B3`](texts/20260905_luna_medium/B3) | Flat-vector-RAG | 将历史正文切成 900-token、重叠 120-token 的扁平片段，以本地 `BAAI/bge-small-zh-v1.5` 编码，按余弦相似度检索 Top-8。 |
| [`B4`](texts/20260905_luna_medium/B4) | Hierarchical-summary | 维护每章最多 600 tokens 的章节摘要和每卷最多 800 tokens 的卷摘要；提示词优先装入全部卷摘要，再装入最近 8 个章节摘要。 |
| [`TaskGraph`](texts/20260905_luna_medium/TaskGraph) | Dynamic TaskGraph memory | 以 L2 卷、L3 章节和 L4 原子状态构图；写作前从动态活跃池、语义检索与已有依赖邻域形成候选，由模型筛选相关 L4→L3 依赖并投影为自然语言事实，同时单独提供上一章正文。写作后抽取并更新具有生命周期的 L4 状态。 |

TaskGraph 的语义编码使用 DashScope `qwen3.7-text-embedding`（2,048 维）；候选池上限
为 192，动态活跃池上限为 100，模型判断相关节点时不设固定入选条数。提交到本仓库的
仅为最终章节正文；运行状态、模型调用记录、图、嵌入索引和失败尝试仍属于外部实验产物，
不与小说文本混放。本目录代表一次完整单重复实验，不应被解释为跨随机种子的方差估计。

## 五基线重跑（正文与中间产物分开）

在项目根目录执行，默认使用 GPT-5.6 Luna / medium，最多重试 5 次：

```powershell
./M5/prepare_baseline_storage.ps1
./M5/protect_codex_history.ps1 -Action Apply
npm install --prefix 'D:\MyCodes\Multi-level_graph\实验产物\cache\codex-cli-0.153.4' --cache 'D:\MyCodes\Multi-level_graph\实验产物\cache\npm' --registry https://registry.npmjs.org --ignore-scripts --no-audit --no-fund --save-exact '@openai/codex@0.153.4'
python -B M5/launch_baselines.py launch --run-id 20260905_luna_medium --chapter-end 320
python -B M5/launch_baselines.py status --run-id 20260905_luna_medium
```

正文存放于 `M5/texts/<run-id>/B0` 至 `B4`；运行记录、失败尝试、
输入包和状态存放于 `D:\MyCodes\Multi-level_graph\实验产物/runs/<run-id>`，
缓存、测试产物与隔离工作目录分别位于该产物根目录的 `cache`、`validation`、`actors`。
每次正式运行都会实测沙箱无法读取金标、后续章节提示词、控制器产物及历史会话。
隔离 actor 必须物理位于仓库外；本机的 `实验产物/actors` 因而是指向仓库外目录的
Windows junction，其余中间产物均物理位于上述 `实验产物` 目录。

六组重生成由 `continue_with_taskgraph.py` 接续：默认等待 B3 成功完成且正文达到
320 章后再启动 TaskGraph；用户明确授权并行时使用 `--start-now` 立即启动。
TaskGraph 的状态与图结构保存在
`实验产物/runs/<run-id>/TaskGraph`，已提交正文同步到
`M5/texts/<run-id>/TaskGraph`。B3 失败、进程消失或 TaskGraph 已有运行记录时，
接续器只记录错误并停止，不会自动重试或覆盖断点。

首次启动可使用 `--chapter-end 1` 完成真实冒烟测试，五组成功退出后再用同一
`run-id` 与 `--chapter-end 320` 继续。已接受的章节保留，章节依次生成；
模型、推理级别、上下文预算、重试上限与正文目录在运行状态中冻结。
失败尝试与调用记录持久保存，重启不会清零同一章的重试预算。
`process_status.json` 记录控制器进程、退出码与异常，`launch.json` 记录精确命令。
启动器核对存活 PID 的进程名、worker/方法脚本及运行目录，避免 Windows 复用旧 PID
后将无关进程误报为仍在运行的控制器。
启动器固定使用产物缓存内的 Codex CLI 0.153.4（用户于 2026-09-06 指定升级），
不修改全局 CLI。安装一次即可；若使用自定义 artifact-root，应在其 cache 内安装
相同版本。CLI 版本属于冻结的隔离证据，不能只替换可执行文件后直接恢复旧断点。
Windows 沙箱的历史保护只列出稳定目录和主数据库；不要把瞬态 `*.sqlite-wal` 或
`*.sqlite-shm` 路径加入 deny 列表，否则缺失路径会被沙箱初始化为目录并阻塞会话库。
全新实验使用新 `run-id`；本次用户明确授权的断点升级按下面的迁移流程执行。
多组恢复宜通过 `--methods B1` 等依次启动，预检通过后再启动下一组。

### 已有断点升级到 0.153.4

确认目标控制器已退出后，逐组执行（示例为 B1）：

```powershell
python -B M5/migrate_cli_version.py --method B1 --run-id 20260905_luna_medium
python -B M5/launch_baselines.py launch --methods B1 --run-id 20260905_luna_medium --chapter-end 320
```

迁移器持有独占运行锁，重新执行真实沙箱隔离测试，只允许 CLI 版本及验证时间变化。
如源码最低版本声明已从 `(0, 138, 0)` 更新为 `(0, 153, 4)`，必须通过仅还原这一行
精确重建旧源码指纹，才允许同步迁移指纹；其他源码变化直接拒绝。
原断点、原隔离证据及变更说明保存在运行目录的 `cli_migrations/to_0.153.4/`。
正文、记忆、历史调用、失败记录与重试预算不会重置；已完成 320 章的 B0 保持历史原样。
迁移前的调用仍属于旧 CLI 阶段，不能把整个混合版本运行报告为全程使用 0.153.4。
迁移检查不调用生成模型；恢复后的下一笔正常章节或摘要调用用于验证新版实际生成。
新版 Windows 沙箱初始化在本机曾超过 60 秒；迁移器和新启动的控制器通过
`cli_preflight_compat.py` 将沙箱预检等待上限设为 300 秒，实际值记入迁移或进程记录。
该兼容层不修改探针、ACL、通过条件，也不改变生成调用的 900 秒超时或重试次数。

在项目根目录安装依赖，然后进入 `M5`：

```powershell
python -m pip install -r ..\requirements.txt
```

正式运行前配置 Codex writer 隔离：

```powershell
powershell -ExecutionPolicy Bypass -File methods/TaskGraph/set_isolation_acl.ps1 -Action Apply
```

分别运行六个实验条件：

```powershell
python methods/B0/run.py run --run-dir methods/B0/run --chapter-end 320
python methods/B1/run.py run --run-dir methods/B1/run --chapter-end 320
python methods/B2/run.py run --run-dir methods/B2/run --chapter-end 320
python methods/B3/run.py run --run-dir methods/B3/run --chapter-end 320
python methods/B4/run.py run --run-dir methods/B4/run --chapter-end 320
python methods/TaskGraph/run.py run --run-dir methods/TaskGraph/run --chapter-end 320
```

B3 运行前准备本地中文嵌入模型。TaskGraph 使用 DashScope 时设置 `DASHSCOPE_API_KEY`，
使用 OpenRouter 时设置 `OPENROUTER_API_KEY` 并添加 `--embedding-provider openrouter`。

查看运行状态：

```powershell
python methods/B0/run.py status --run-dir methods/B0/run
python methods/TaskGraph/run.py status --run-dir methods/TaskGraph/run
```

六个条件完成后生成评审包并汇总评审结果：

```powershell
python evaluation/prepare_packets.py
python evaluation/aggregate_results.py
```
