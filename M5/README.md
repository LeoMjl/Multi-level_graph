# M5 运行方法

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
