# M5 提示词运行方法

提示词由各实验入口自动读取，无需单独执行。运行任一条件时，入口会加载全局任务、故事设定、
对应章节提示和该条件允许使用的记忆上下文：

```powershell
python methods/B0/run.py run --run-dir methods/B0/run --chapter-end 320
python methods/TaskGraph/run.py run --run-dir methods/TaskGraph/run --chapter-end 320
```

完成全部条件后，评审流程会读取模型评审与矛盾复核提示：

```powershell
python evaluation/prepare_packets.py
python evaluation/aggregate_results.py
```
