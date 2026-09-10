"""Launch five independent, resumable M5 controllers with external artifacts."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

M5 = Path(__file__).resolve().parent
sys.path.insert(0, str(M5 / 'methods/B0/src'))
from mlg.m5.io import atomic_write_json, read_json
from controller_process_identity import controller_identity_matches


def process_alive(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def runtime_environment(artifact):
    env = dict(os.environ)
    cache = artifact / 'cache'
    paths = {'HF_HOME': cache / 'huggingface', 'HF_HUB_CACHE': cache / 'huggingface/hub',
             'TIKTOKEN_CACHE_DIR': cache / 'tiktoken', 'TMP': cache / 'tmp',
             'TEMP': cache / 'tmp', 'TORCH_HOME': cache / 'torch',
             'XDG_CACHE_HOME': cache / 'xdg', 'PYTHONPYCACHEPREFIX': cache / 'pycache'}
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    env.pop('PYTHONPATH', None)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['launch', 'status'])
    parser.add_argument('--artifact-root', type=Path, default=M5 / 'artifacts')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--chapter-end', type=int, default=320)
    parser.add_argument('--methods', nargs='+', choices=['B0','B1','B2','B3','B4'], default=['B0','B1','B2','B3','B4'])
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {'.','..'}:
        raise ValueError('run-id must be one directory name')
    artifact = args.artifact_root.resolve()
    run_root = artifact / 'runs' / args.run_id
    env = runtime_environment(artifact) if args.action == 'launch' else None
    rows = []
    for method in args.methods:
        run_dir = run_root / method
        text_dir = M5 / 'texts' / args.run_id / method
        process_file = run_dir / 'process_status.json'
        prior = read_json(process_file) if process_file.exists() else {}
        alive = (prior.get('status') not in {'failed', 'completed'}
                 and process_alive(prior['pid'])) if prior.get('pid') else False
        if alive:
            alive = controller_identity_matches(
                prior['pid'], M5 / 'baseline_worker.py',
                M5 / 'methods' / method / 'run.py', run_dir)
        if args.action == 'launch':
            if alive:
                raise RuntimeError(f'{method} controller {prior["pid"]} is still running')
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-B', str(M5 / 'baseline_worker.py'),
                       str(M5 / 'methods' / method / 'run.py'), 'run',
                       '--run-dir', str(run_dir), '--text-dir', str(text_dir),
                       '--formal-isolation-root', str(artifact / 'actors'),
                       '--m5-root', str(M5), '--run-mode', 'formal', '--replicate', '1',
                       '--chapter-end', str(args.chapter_end), '--model', 'gpt-5.6-luna',
                       '--reasoning-effort', 'medium', '--token-budget', '12000',
                       '--retries', '5', '--timeout-seconds', '900']
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            log_path = run_dir / f'controller_{stamp}.log'
            with log_path.open('ab', buffering=0) as log:
                process = subprocess.Popen(command, cwd=M5, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            prior = {'pid': process.pid, 'status': 'starting', 'log': str(log_path),
                     'command': command, 'text_dir': str(text_dir)}
            atomic_write_json(run_dir / 'launch.json', prior)
            alive = True
        state_file = run_dir / 'state.json'
        state = read_json(state_file) if state_file.exists() else {}
        rows.append({'method': method, 'pid': prior.get('pid'), 'process_alive': alive,
                     'status': prior.get('status', 'not_started'),
                     'last_completed': state.get('last_completed', 0),
                     'stage': state.get('stage'), 'text_dir': str(text_dir),
                     'run_dir': str(run_dir), 'error': prior.get('error')})
    print(json.dumps(rows, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
