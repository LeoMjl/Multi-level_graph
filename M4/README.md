# M4 Running Instructions

Open PowerShell in the `M4` directory and install the shared dependencies:

```powershell
python -m pip install -r ..\requirements.txt
```

Place the QuALITY and MultiHop-RAG datasets in the following directories:

```text
data/raw/quality/
data/raw/multihop_rag/
```

Run all datasets and methods:

```powershell
python run.py --model-config model_config.example.json
```

Select datasets, methods, or a sample limit:

```powershell
python run.py --model-config model_config.example.json `
  --datasets quality multihop_rag `
  --methods full_text raptor graphrag memorystream taskgraph `
  --limit 20
```

Use `--build-only` to build indexes, `--judge-only` to run evaluation, and
`--report-only` to generate the summary. Set `GRAPHRAG_ROOT` to specify the
GraphRAG source directory.
