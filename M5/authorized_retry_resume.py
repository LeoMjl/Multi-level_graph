"""Apply explicit per-chapter retry amendments without changing baseline inputs."""
from __future__ import annotations

from copy import copy
from pathlib import Path


def install_retry_extension(run_dir: Path):
    from mlg.m5 import baseline_runtime
    from mlg.m5.collab_state import CollaborationProtocolError
    from mlg.m5.io import atomic_write_json, read_json

    run_dir = run_dir.resolve()
    authorization_paths = [run_dir / 'retry_authorization.json']
    authorization_paths.extend(sorted(run_dir.glob('retry_authorization_*.json')))
    authorization_paths = [path for path in authorization_paths if path.is_file()]
    if not authorization_paths:
        return None
    authorizations = []
    extensions = {}
    for authorization_path in authorization_paths:
        authorization = read_json(authorization_path)
        if (authorization.get('schema') != 'm5-user-retry-amendment-v1'
                or Path(authorization.get('run_dir', '')).resolve() != run_dir
                or authorization.get('authorized_by') != 'user'
                or authorization.get('base_retries') != 5):
            raise CollaborationProtocolError('Invalid retry amendment authorization')
        targets = authorization.get('targets', [])
        if not isinstance(targets, list) or not targets:
            raise CollaborationProtocolError('Retry amendment has no explicit targets')
        seen = set()
        for target in targets:
            chapter = target.get('chapter_id')
            extra = target.get('extra_attempts')
            if (type(chapter) is not int or not 1 <= chapter <= 320
                    or target.get('kind') != 'writer' or type(extra) is not int
                    or extra != 5 or chapter in seen):
                raise CollaborationProtocolError('Invalid or duplicate retry target')
            seen.add(chapter)
            extensions.setdefault(chapter, []).append(
                (authorization_path, authorization, extra))
        authorizations.append(authorization)

    original = baseline_runtime.validated_generation

    def authorized_generation(run, writer, args, chapter_id, request, kind):
        grants = extensions.get(chapter_id, []) if kind == 'writer' else []
        if not grants:
            return original(run, writer, args, chapter_id, request, kind)
        if Path(run.run_dir).resolve() != run_dir or args.retries != 5:
            raise CollaborationProtocolError('Retry amendment does not match this run')
        total_attempts = args.retries + 1
        total_extra = 0
        for source, authorization, extra in grants:
            applied = {
                'schema': 'm5-applied-retry-amendment-v1',
                'authorization': authorization,
                'chapter_id': chapter_id, 'kind': kind,
                'base_total_attempts': total_attempts,
                'extra_attempts': extra,
                'effective_total_attempts': total_attempts + extra,
            }
            applied_path = (
                run_dir / f'chapter_{chapter_id:03d}_retry_amendment.json'
                if source.name == 'retry_authorization.json' else
                run_dir / f'chapter_{chapter_id:03d}_{source.stem}.applied.json'
            )
            if applied_path.exists() and read_json(applied_path) != applied:
                raise CollaborationProtocolError('Previously applied retry amendment changed')
            if not applied_path.exists():
                atomic_write_json(applied_path, applied)
            total_attempts += extra
            total_extra += extra
        # Only the invocation budget changes. The baseline configuration, prompt,
        # existing attempt receipts and validation gates remain untouched.
        call_args = copy(args)
        call_args.retries += total_extra
        return original(run, writer, call_args, chapter_id, request, kind)

    baseline_runtime.validated_generation = authorized_generation
    if len(authorizations) == 1:
        return authorizations[0]
    return {'schema': 'm5-user-retry-amendment-set-v1',
            'authorizations': authorizations}
