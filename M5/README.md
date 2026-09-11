# M5 Running Instructions

Install the dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Enter the `M5` directory and configure isolation permissions for the Codex writer:

```powershell
Set-Location M5
powershell -ExecutionPolicy Bypass -File methods/TaskGraph/set_isolation_acl.ps1 -Action Apply
```

Run the six experimental conditions:

```powershell
python methods/B0/run.py run --run-dir methods/B0/run --chapter-end 320
python methods/B1/run.py run --run-dir methods/B1/run --chapter-end 320
python methods/B2/run.py run --run-dir methods/B2/run --chapter-end 320
python methods/B3/run.py run --run-dir methods/B3/run --chapter-end 320
python methods/B4/run.py run --run-dir methods/B4/run --chapter-end 320
python methods/TaskGraph/run.py run --run-dir methods/TaskGraph/run --chapter-end 320
```

Prepare a local Chinese embedding model before running B3. For TaskGraph, set
`DASHSCOPE_API_KEY` when using DashScope. To use OpenRouter, set
`OPENROUTER_API_KEY` and add `--embedding-provider openrouter` to the command.

Check run status:

```powershell
python methods/B0/run.py status --run-dir methods/B0/run
python methods/TaskGraph/run.py status --run-dir methods/TaskGraph/run
```

After all six conditions finish, prepare the review packets and aggregate the results:

```powershell
python evaluation/prepare_packets.py
python evaluation/aggregate_results.py
```
