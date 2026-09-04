# M4 运行方法

在项目根目录安装依赖，然后进入 `M4`：

```powershell
python -m pip install -r ..\requirements.txt
```

将 QuALITY 和 MultiHop-RAG 数据分别放入：

```text
data/raw/quality/
data/raw/multihop_rag/
```

运行全部数据集和方法：

```powershell
python run.py --model-config model_config.example.json
```

指定数据集、方法或样本数：

```powershell
python run.py --model-config model_config.example.json `
  --datasets quality multihop_rag `
  --methods full_text raptor graphrag memorystream taskgraph `
  --limit 20
```

使用 `--build-only` 仅构建索引，使用 `--judge-only` 运行评审，使用 `--report-only` 生成汇总。
GraphRAG 源码目录可通过 `GRAPHRAG_ROOT` 环境变量指定。
