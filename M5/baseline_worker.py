"""Run one baseline and persist a durable process exit status."""
from __future__ import annotations

import json
import os
import runpy
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def main():
    script = Path(sys.argv[1]).resolve()
    arguments = sys.argv[2:]
    run_dir = Path(arguments[arguments.index('--run-dir') + 1])
    sys.path.insert(0, str(script.parent / 'src'))
    from mlg.m5.io import atomic_write_json

    status_path = run_dir / 'process_status.json'
    status = {
        'pid': os.getpid(), 'script': str(script), 'arguments': arguments,
        'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
    }
    atomic_write_json(status_path, status)
    sys.argv = [str(script), *arguments]
    exit_code = 0
    try:
        from cli_preflight_compat import configure_preflight_timeout
        status['preflight_timeout_seconds'] = configure_preflight_timeout()
        atomic_write_json(status_path, status)
        from authorized_retry_resume import install_retry_extension
        amendment = install_retry_extension(run_dir)
        if amendment is not None:
            status['retry_amendment'] = amendment
            atomic_write_json(status_path, status)
        runpy.run_path(str(script), run_name='__main__')
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        if exit_code:
            status['error'] = str(exc)
    except BaseException as exc:
        exit_code = 1
        status['error'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        status.update(status='completed' if exit_code == 0 else 'failed',
                      exit_code=exit_code, finished_at=datetime.now(timezone.utc).isoformat())
        atomic_write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=True), flush=True)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
