# Running the M5 Conditions

Run the following commands from the `M5` directory:

```powershell
python methods/B0/run.py run --run-dir methods/B0/run --chapter-end 320
python methods/B1/run.py run --run-dir methods/B1/run --chapter-end 320
python methods/B2/run.py run --run-dir methods/B2/run --chapter-end 320
python methods/B3/run.py run --run-dir methods/B3/run --chapter-end 320
python methods/B4/run.py run --run-dir methods/B4/run --chapter-end 320
python methods/TaskGraph/run.py run --run-dir methods/TaskGraph/run --chapter-end 320
```

Use `status` to inspect progress and `preflight` to check the runtime requirements.
Resume an interrupted TaskGraph run with:

```powershell
python methods/TaskGraph/run.py resume --run-dir methods/TaskGraph/run --chapter-end 320
```
