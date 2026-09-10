"""Reject stale controller PIDs that Windows has reused for another process."""
import json
import subprocess
from pathlib import Path


def matches_controller(process, worker, script, run_dir):
    if not process:
        return False
    name = str(process.get('Name', '')).casefold()
    command = str(process.get('CommandLine', '')).casefold().replace('/', '\\')
    return name in {'python.exe', 'pythonw.exe'} and all(
        str(Path(path).resolve()).casefold().replace('/', '\\') in command
        for path in (worker, script, run_dir)
    )


def controller_identity_matches(pid, worker, script, run_dir):
    query = (
        f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | "
        "Select-Object Name,CommandLine | ConvertTo-Json -Compress"
    )
    output = subprocess.check_output(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
         '[Console]::OutputEncoding=[Text.Encoding]::UTF8; ' + query],
        text=True, encoding='utf-8-sig', timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).strip()
    process = json.loads(output) if output else None
    if process and not process.get('CommandLine'):
        raise RuntimeError('Cannot verify controller process command line; refusing duplicate launch')
    return matches_controller(process, worker, script, run_dir)
