from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from mlg.config import RAW_DIR, ensure_runtime_dirs
from mlg.data.sources import DataSource, select_sources


def fetch_datasets(dataset: str = "all", version: str = "classic") -> list[dict[str, Any]]:
    ensure_runtime_dirs()
    manifests = []
    for source in select_sources(dataset, version):
        manifests.append(fetch_source(source))
    return manifests


def fetch_source(source: DataSource) -> dict[str, Any]:
    repo_root = RAW_DIR / "repos"
    repo_root.mkdir(parents=True, exist_ok=True)
    target = repo_root / source.repo_dir
    if source.name == "longmemeval_v2":
        return fetch_longmemeval_v2_official(source, target)
    if source.name == "locomo":
        return fetch_locomo_official(source, target)
    if source.name == "stabletoolbench":
        return fetch_stabletoolbench_official(source, target)
    if source.hf_dataset_id:
        return fetch_hf_source(source, target)

    status = "ok"
    error = ""
    if target.exists() and (target / ".git").exists():
        cmd = ["git", "-C", str(target), "fetch", "--depth", "1", "origin", source.preferred_revision]
        checkout = ["git", "-C", str(target), "checkout", "FETCH_HEAD"]
    else:
        cmd = ["git", "clone", "--depth", "1", "--branch", source.preferred_revision, source.url, str(target)]
        checkout = []
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if checkout:
            subprocess.run(checkout, check=True, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - network dependent
        status = "error"
        error = str(exc)

    revision = ""
    if (target / ".git").exists():
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            revision = ""
    manifest = {
        "dataset": source.name,
        "version_group": source.version_group,
        "source_url": source.url,
        "preferred_revision": source.preferred_revision,
        "resolved_revision": revision,
        "local_path": str(target),
        "status": status,
        "error": error,
        "license_note": source.license_note,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tree_checksum": directory_checksum(target) if target.exists() else "",
    }
    manifest_path = RAW_DIR / "manifests" / f"{source.name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def fetch_locomo_official(source: DataSource, target: Path) -> dict[str, Any]:
    """Clone the official release and verify its annotated benchmark file."""
    repository = target / "repository"
    status = "ok"
    error = ""
    revision = ""
    dataset_file = repository / "data" / "locomo10.json"
    try:
        target.mkdir(parents=True, exist_ok=True)
        if (repository / ".git").is_dir():
            subprocess.run(
                ["git", "-C", str(repository), "fetch", "--depth", "1", "origin", source.preferred_revision],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "checkout", "FETCH_HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    source.preferred_revision,
                    source.url,
                    str(repository),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if not dataset_file.is_file():
            raise FileNotFoundError(f"Official LoCoMo annotations are missing: {dataset_file}")
        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != 10:
            raise ValueError("Official LoCoMo release must contain 10 conversations")
    except Exception as exc:  # pragma: no cover - network dependent
        status = "error"
        error = str(exc)

    manifest = {
        "dataset": source.name,
        "version_group": source.version_group,
        "source_url": source.url,
        "data_source_type": "official_repository",
        "preferred_revision": source.preferred_revision,
        "resolved_revision": revision,
        "local_path": str(target),
        "repository_path": str(repository),
        "dataset_path": str(dataset_file),
        "status": status,
        "error": error,
        "license_note": source.license_note,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tree_checksum": file_checksum(dataset_file) if dataset_file.is_file() else "",
    }
    manifest_path = RAW_DIR / "manifests" / f"{source.name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def fetch_stabletoolbench_official(
    source: DataSource,
    target: Path,
) -> dict[str, Any]:
    """Clone and validate the official 765-query StableToolBench release."""
    repository = target / "repository"
    status = "ok"
    error = ""
    revision = ""
    warnings: list[str] = []
    subsets = {
        "G1_instruction": 163,
        "G1_category": 153,
        "G1_tool": 158,
        "G2_instruction": 106,
        "G2_category": 124,
        "G3_instruction": 61,
    }
    counts: dict[str, int] = {}
    try:
        target.mkdir(parents=True, exist_ok=True)
        if not (repository / ".git").is_dir():
            repository.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "-C", str(repository), "init"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "remote", "add", "origin", source.url],
                check=True,
                capture_output=True,
                text=True,
            )
        try:
            subprocess.run(
                [
                    "git", "-C", str(repository), "fetch", "--depth", "1",
                    "origin", source.preferred_revision,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            if not (repository / "solvable_queries").is_dir():
                raise
            warnings.append(
                "remote refresh failed; validated the existing official checkout: "
                + (exc.stderr or str(exc)).strip()
            )
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        query_root = repository / "solvable_queries"
        for subset, expected in subsets.items():
            query_path = query_root / "test_instruction" / f"{subset}.json"
            id_path = query_root / "test_query_ids" / f"{subset}.json"
            rows = json.loads(query_path.read_text(encoding="utf-8"))
            ids = json.loads(id_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not isinstance(ids, dict):
                raise ValueError(f"Invalid official StableToolBench files for {subset}")
            counts[subset] = len(rows)
            row_ids = {str(row.get("query_id", "")) for row in rows}
            if len(rows) != expected or row_ids != set(ids):
                raise ValueError(
                    f"StableToolBench {subset} expected {expected} aligned queries"
                )
        if sum(counts.values()) != 765:
            raise ValueError("Official StableToolBench must contain 765 solvable queries")
    except Exception as exc:  # pragma: no cover - network dependent
        status = "error"
        error = str(exc)

    query_root = repository / "solvable_queries"
    manifest = {
        "dataset": source.name,
        "version_group": source.version_group,
        "source_url": source.url,
        "data_source_type": "official_repository",
        "preferred_revision": source.preferred_revision,
        "resolved_revision": revision,
        "local_path": str(target),
        "repository_path": str(repository),
        "dataset_path": str(query_root),
        "status": status,
        "error": error,
        "warnings": warnings,
        "license_note": source.license_note,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "subset_counts": counts,
        "episode_count": sum(counts.values()),
        "official_metrics": ["SoPR", "SoWR", "FAC"],
        "tree_checksum": directory_checksum(query_root) if query_root.exists() else "",
    }
    manifest_path = RAW_DIR / "manifests" / f"{source.name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def fetch_longmemeval_v2_official(source: DataSource, target: Path) -> dict[str, Any]:
    """Use the release repository's downloader instead of guessing Hub filenames."""
    repository = target / "repository"
    dataset = target / "dataset"
    status = "ok"
    error = ""
    revision = ""
    try:
        target.mkdir(parents=True, exist_ok=True)
        if (repository / ".git").is_dir():
            subprocess.run(
                ["git", "-C", str(repository), "fetch", "--depth", "1", "origin", source.preferred_revision],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "checkout", "FETCH_HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    source.preferred_revision,
                    source.url,
                    str(repository),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        download_script = repository / "data" / "download_data.py"
        if not download_script.is_file():
            raise FileNotFoundError(f"Official LongMemEval-V2 downloader is missing: {download_script}")
        subprocess.run(
            [sys.executable, str(download_script), "--data-root", str(dataset)],
            check=True,
        )
        required = (
            dataset / "questions.jsonl",
            dataset / "trajectories.jsonl",
            dataset / "haystacks" / "lme_v2_small.json",
            dataset / "haystacks" / "lme_v2_medium.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Official LongMemEval-V2 download is incomplete: {missing}")
    except Exception as exc:  # pragma: no cover - network dependent
        status = "error"
        error = str(exc)

    checksum_manifest = dataset / "checksums.sha256"
    manifest = {
        "dataset": source.name,
        "version_group": source.version_group,
        "source_url": source.url,
        "data_source_type": "official_repository_downloader",
        "hf_dataset_id": source.hf_dataset_id,
        "preferred_revision": source.preferred_revision,
        "resolved_revision": revision,
        "local_path": str(target),
        "repository_path": str(repository),
        "dataset_path": str(dataset),
        "official_download_script": str(repository / "data" / "download_data.py"),
        "status": status,
        "error": error,
        "license_note": source.license_note,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tree_checksum": file_checksum(checksum_manifest) if checksum_manifest.is_file() else "",
        "next": [
            f"{sys.executable} {repository / 'data' / 'prepare_data.py'} --data-root {dataset} --mode symlink",
            f"{sys.executable} {repository / 'data' / 'validate_data.py'} --data-root {dataset} --tier small",
        ],
    }
    manifest_path = RAW_DIR / "manifests" / f"{source.name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def fetch_hf_source(source: DataSource, target: Path) -> dict[str, Any]:
    status = "ok"
    error = ""
    revision = ""
    try:
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().dataset_info(source.hf_dataset_id, files_metadata=False)
        revision = info.sha or ""
        target.mkdir(parents=True, exist_ok=True)
        failures = []
        for sibling in info.siblings:
            filename = sibling.rfilename
            if source.include_files and filename not in source.include_files:
                continue
            if not is_allowed_dataset_file(filename):
                continue
            try:
                hf_hub_download(
                    repo_id=source.hf_dataset_id,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                )
            except Exception as file_exc:
                failures.append({"file": filename, "error": str(file_exc)})
        extract_archives(target)
        if failures:
            status = "partial"
            error = json.dumps(failures, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - network dependent
        status = "error"
        error = str(exc)

    manifest = {
        "dataset": source.name,
        "version_group": source.version_group,
        "source_url": source.url,
        "data_source_type": "huggingface_dataset",
        "hf_dataset_id": source.hf_dataset_id,
        "preferred_revision": source.preferred_revision,
        "resolved_revision": revision,
        "local_path": str(target),
        "status": status,
        "error": error,
        "license_note": source.license_note,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tree_checksum": directory_checksum(target) if target.exists() else "",
    }
    manifest_path = RAW_DIR / "manifests" / f"{source.name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def directory_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    if not path.exists():
        return ""
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file() or ".git" in file_path.parts:
            continue
        rel = file_path.relative_to(path).as_posix()
        hasher.update(rel.encode("utf-8"))
        try:
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError:
            continue
    return hasher.hexdigest()


def file_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_archives(path: Path) -> None:
    for archive in path.rglob("*.zip"):
        target_dir = archive.with_suffix("")
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                destination = target_dir.resolve()
                for member in zf.infolist():
                    member_target = (target_dir / member.filename).resolve()
                    if destination != member_target and destination not in member_target.parents:
                        raise ValueError(f"Refusing unsafe archive member path: {member.filename}")
            zf.extractall(target_dir)
        except zipfile.BadZipFile:
            continue


def is_allowed_dataset_file(filename: str) -> bool:
    allowed_suffixes = (".json", ".jsonl", ".csv", ".parquet", ".zip", ".py", ".md")
    name = filename.rsplit("/", 1)[-1]
    return (
        filename.endswith(allowed_suffixes)
        or name.startswith("longmemeval_")
        or name.upper().startswith("README")
    )
