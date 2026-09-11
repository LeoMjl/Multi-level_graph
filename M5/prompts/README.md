# Running M5 with the Prompt Files

The experiment entry points load the prompt files automatically. When a condition is
run, its entry point loads the global task, story setting, chapter prompt, and the
memory context permitted for that condition:

```powershell
python methods/B0/run.py run --run-dir methods/B0/run --chapter-end 320
python methods/TaskGraph/run.py run --run-dir methods/TaskGraph/run --chapter-end 320
```

After all conditions finish, run the review pipeline to load the model-review and
contradiction-check prompts:

```powershell
python evaluation/prepare_packets.py
python evaluation/aggregate_results.py
```
