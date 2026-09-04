# M3 运行方法

安装并准备 StableToolBench：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_dataset.ps1
```

在带有 Docker 和 GPU 的 Bash 环境中配置脚本所需的模型目录与环境变量，然后启动服务：

```bash
bash scripts/start_official_services.sh
```

运行 TaskGraph、CoT 或 DFS：

```bash
METHOD=ours SUBSET=all bash scripts/run_toolbench.sh
METHOD=cot SUBSET=all bash scripts/run_toolbench.sh
METHOD=dfs SUBSET=all bash scripts/run_toolbench.sh
```

运行官方评估：

```bash
METHOD=ours bash scripts/evaluate_official.sh
METHOD=cot bash scripts/evaluate_official.sh
METHOD=dfs bash scripts/evaluate_official.sh
```

服务地址、模型目录、并发数、样本范围和凭据均通过脚本顶部列出的环境变量设置。
