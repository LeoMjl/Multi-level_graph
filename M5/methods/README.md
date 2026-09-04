# M5 各条件运行方法

从 `M5` 目录运行：

```powershell
python methods/B0/run.py run --run-dir methods/B0/run --chapter-end 320
python methods/B1/run.py run --run-dir methods/B1/run --chapter-end 320
python methods/B2/run.py run --run-dir methods/B2/run --chapter-end 320
python methods/B3/run.py run --run-dir methods/B3/run --chapter-end 320
python methods/B4/run.py run --run-dir methods/B4/run --chapter-end 320
python methods/TaskGraph/run.py run --run-dir methods/TaskGraph/run --chapter-end 320
```

使用 `status` 查看进度，使用 `preflight` 检查运行条件。TaskGraph 中断后使用 `resume` 继续：

```powershell
python methods/TaskGraph/run.py resume --run-dir methods/TaskGraph/run --chapter-end 320
```
