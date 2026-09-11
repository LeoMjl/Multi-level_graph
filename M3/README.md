# M3 Running Instructions

Open PowerShell in the `M3` directory and install StableToolBench:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_dataset.ps1
```

In a Bash environment with Docker and GPU support, configure the model directories
and environment variables listed in the service script, then start the services:

```bash
bash scripts/start_official_services.sh
```

Run TaskGraph, CoT, or DFS:

```bash
METHOD=ours SUBSET=all bash scripts/run_toolbench.sh
METHOD=cot SUBSET=all bash scripts/run_toolbench.sh
METHOD=dfs SUBSET=all bash scripts/run_toolbench.sh
```

Run the official evaluation:

```bash
METHOD=ours bash scripts/evaluate_official.sh
METHOD=cot bash scripts/evaluate_official.sh
METHOD=dfs bash scripts/evaluate_official.sh
```

Set service endpoints, model directories, concurrency, sample ranges, and credentials
through the environment variables listed at the top of each script.
