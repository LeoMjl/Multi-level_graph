# M2 Running Instructions

Open PowerShell in the `M2` directory, install the shared dependencies, and prepare the dataset:

```powershell
python -m pip install -r ..\requirements.txt
python run.py fetch
python run.py prepare
python run.py run --model-config model_config.example.json
```

Use `--limit` to restrict the number of samples, `--methods` to select methods, and
`--run-dir` to choose the output directory:

```powershell
python run.py run --model-config model_config.example.json --methods ours --limit 20 --run-dir results
```

Configure the model endpoint, model name, and API-key environment variable in
`model_config.example.json`.
