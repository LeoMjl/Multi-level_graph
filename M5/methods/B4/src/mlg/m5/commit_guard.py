from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from .dataset import han_char_count
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mlg.m5.dataset import han_char_count


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing:{path.as_posix()}")


def guard_commit(run_root: Path, chapter: int, repair: bool) -> dict[str, Any]:
    errors: list[str] = []
    stem = f"chapter_{chapter:03d}"
    paths = {
        "chapter": run_root / "chapters" / f"{stem}.md",
        "input": run_root / "inputs" / f"{stem}.json",
        "record": run_root / "records" / f"{stem}.json",
        "checkpoint": run_root / "checkpoints" / f"{stem}.json",
        "state": run_root / "memory" / f"state_through_{chapter:03d}.md",
        "delta": run_root / "graph" / "deltas" / f"{stem}.json",
    }
    for path in paths.values():
        _require(path, errors)
    if errors:
        return {"ok": False, "chapter": chapter, "errors": errors}

    text_bytes = paths["chapter"].read_bytes()
    text = text_bytes.decode("utf-8")
    body = "\n".join(text.splitlines()[1:])
    actual = {
        "han_chars": han_char_count(text),
        "han_chars_body_no_title": han_char_count(body),
        "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
    }
    if not 2000 <= actual["han_chars_body_no_title"] <= 3000:
        errors.append(f"body_han_out_of_range:{actual['han_chars_body_no_title']}")

    record = _read_json(paths["record"])
    checkpoint = _read_json(paths["checkpoint"])
    delta = _read_json(paths["delta"])
    for label, artifact in (("record", record), ("checkpoint", checkpoint)):
        for field, value in actual.items():
            if artifact.get(field) != value:
                errors.append(f"{label}_{field}_mismatch")
    expected_state = f"memory/state_through_{chapter:03d}.md"
    if record.get("source_text") != f"chapters/{stem}.md":
        errors.append("record_source_text_mismatch")
    if record.get("state_source") != expected_state:
        errors.append("record_state_source_mismatch")
    if checkpoint.get("state_path") != expected_state:
        errors.append("checkpoint_state_path_mismatch")
    if checkpoint.get("record_path") != f"records/{stem}.json":
        errors.append("checkpoint_record_path_mismatch")

    for number in range(1, chapter):
        prior_stem = f"chapter_{number:03d}"
        prior_text_path = run_root / "chapters" / f"{prior_stem}.md"
        prior_record_path = run_root / "records" / f"{prior_stem}.json"
        prior_checkpoint_path = run_root / "checkpoints" / f"{prior_stem}.json"
        prior_state_path = run_root / "memory" / f"state_through_{number:03d}.md"
        for prior_path in (
            prior_text_path,
            prior_record_path,
            prior_checkpoint_path,
            prior_state_path,
        ):
            _require(prior_path, errors)
        if not all(
            path.is_file()
            for path in (prior_text_path, prior_record_path, prior_checkpoint_path)
        ):
            continue
        prior_bytes = prior_text_path.read_bytes()
        prior_text = prior_bytes.decode("utf-8")
        prior_actual = {
            "han_chars": han_char_count(prior_text),
            "han_chars_body_no_title": han_char_count(
                "\n".join(prior_text.splitlines()[1:])
            ),
            "text_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        }
        if not 2000 <= prior_actual["han_chars_body_no_title"] <= 3000:
            errors.append(
                f"chapter_{number:03d}_body_han_out_of_range:"
                f"{prior_actual['han_chars_body_no_title']}"
            )
        for label, artifact_path in (
            ("record", prior_record_path),
            ("checkpoint", prior_checkpoint_path),
        ):
            artifact = _read_json(artifact_path)
            for field, value in prior_actual.items():
                if artifact.get(field) != value:
                    errors.append(f"chapter_{number:03d}_{label}_{field}_mismatch")
        prior_record = _read_json(prior_record_path)
        prior_checkpoint = _read_json(prior_checkpoint_path)
        expected_prior_state = f"memory/state_through_{number:03d}.md"
        if prior_record.get("source_text") != f"chapters/{prior_stem}.md":
            errors.append(f"chapter_{number:03d}_record_source_text_mismatch")
        if prior_record.get("state_source") != expected_prior_state:
            errors.append(f"chapter_{number:03d}_record_state_source_mismatch")
        if prior_checkpoint.get("state_path") != expected_prior_state:
            errors.append(f"chapter_{number:03d}_checkpoint_state_path_mismatch")
        if prior_checkpoint.get("record_path") != f"records/{prior_stem}.json":
            errors.append(f"chapter_{number:03d}_checkpoint_record_path_mismatch")

    base = _read_json(run_root / "graph" / "graph.json")
    nodes = list(base.get("nodes", []))
    edges = list(base.get("edges", []))
    historical_ids = {node["node_id"] for node in nodes}
    node_lookup = {node["node_id"]: node for node in nodes}
    expected_deltas: list[str] = []
    for number in range(2, chapter + 1):
        relative = f"graph/deltas/chapter_{number:03d}.json"
        delta_path = run_root / relative
        _require(delta_path, errors)
        if not delta_path.is_file():
            continue
        current = _read_json(delta_path)
        current_nodes = list(current.get("nodes", []))
        current_edges = list(current.get("edges", []))
        node_updates = list(current.get("node_updates", []))
        for update in node_updates:
            node_id = update.get("node_id")
            target = node_lookup.get(node_id)
            if target is None or target.get("level") != "L2":
                errors.append(f"chapter_{number:03d}_invalid_node_update:{node_id}")
                continue
            changes = dict(update.get("set", {}))
            metadata = dict(target.get("metadata", {}))
            metadata.update(changes.pop("metadata", {}))
            target.update(changes)
            target["metadata"] = metadata
        volume_id = ((number - 1) // 40) + 1
        volume_node_id = f"L2_Volume{volume_id}"
        if number > 1 and (number - 1) % 40 == 0:
            activation = node_lookup.get(volume_node_id, {})
            if activation.get("status") != "Active":
                errors.append(f"chapter_{number:03d}_volume_not_activated")
        if number % 40 == 0:
            completed = node_lookup.get(volume_node_id, {})
            completed_metadata = completed.get("metadata", {})
            if completed.get("status") != "Done":
                errors.append(f"chapter_{number:03d}_volume_not_done")
            if completed_metadata.get("completed_chapter") != number:
                errors.append(f"chapter_{number:03d}_volume_completion_not_recorded")
        l3_ids = {node["node_id"] for node in current_nodes if node.get("level") == "L3"}
        l4_ids = {node["node_id"] for node in current_nodes if node.get("level") == "L4"}
        inclusion_pairs = {
            (edge.get("source_id"), edge.get("target_id"))
            for edge in current_edges
            if edge.get("edge_type") == "INCLUSION"
        }
        expected_inclusions = {
            (f"L2_Volume{((number - 1) // 40) + 1}", l3_id) for l3_id in l3_ids
        }
        expected_inclusions.update((l3_id, l4_id) for l3_id in l3_ids for l4_id in l4_ids)
        if len(l3_ids) != 1 or inclusion_pairs != expected_inclusions:
            errors.append(f"chapter_{number:03d}_inclusion_topology_mismatch")
        dependency_sources: list[str] = []
        for edge in current_edges:
            if edge.get("edge_type") != "DEPENDENCY":
                continue
            source = edge.get("source_id")
            dependency_sources.append(source)
            if source not in historical_ids:
                errors.append(
                    f"chapter_{number:03d}_dependency_source_not_historical:{source}"
                )
            if source == "L1_Novel1":
                errors.append(f"chapter_{number:03d}_dependency_source_is_L1")
            if edge.get("target_id") not in (l3_ids | l4_ids):
                errors.append(
                    f"chapter_{number:03d}_dependency_target_is_not_current_node"
                )
        if number == chapter:
            expected_sources = sorted(set(dependency_sources))
            for label, artifact in (("record", record), ("checkpoint", checkpoint)):
                if sorted(set(artifact.get("dependency_sources", []))) != expected_sources:
                    errors.append(f"{label}_dependency_sources_mismatch")
                if artifact.get("dependency_edges") != len(dependency_sources):
                    errors.append(f"{label}_dependency_edges_mismatch")
        nodes.extend(current_nodes)
        edges.extend(current_edges)
        historical_ids.update(node["node_id"] for node in current_nodes)
        node_lookup.update({node["node_id"]: node for node in current_nodes})
        expected_deltas.append(relative)

    node_levels = Counter(node.get("level") for node in nodes)
    edge_types = Counter(edge.get("edge_type") for edge in edges)
    totals = {
        "node_totals": {
            "L1": node_levels["L1"],
            "L2": node_levels["L2"],
            "L3": node_levels["L3"],
            "L4": node_levels["L4"],
            "total": len(nodes),
        },
        "edge_totals": {
            "INCLUSION": edge_types["INCLUSION"],
            "DEPENDENCY": edge_types["DEPENDENCY"],
            "MAINLINE": edge_types["MAINLINE"],
            "total": len(edges),
        },
    }
    if totals["edge_totals"]["MAINLINE"] != 0:
        errors.append("mainline_edges_present")

    index_path = run_root / "graph" / "index.json"
    index = _read_json(index_path)
    expected_index = dict(index)
    expected_index.update(totals)
    expected_index.update(
        {
            "applied_deltas": expected_deltas,
            "last_completed": chapter,
            "checkpoint": f"checkpoints/{stem}.json",
            "future_chapters_materialized": False,
            "volume_statuses": {
                node_id: node.get("status")
                for node_id, node in sorted(node_lookup.items())
                if node.get("level") == "L2"
            },
        }
    )
    index_mismatch = index != expected_index
    did_repair = False
    if index_mismatch and repair and not errors:
        _write_json_atomic(index_path, expected_index)
        index = expected_index
        index_mismatch = False
        did_repair = True
    if index_mismatch:
        errors.append("graph_index_mismatch")

    manifest_path = run_root.parent.parent / "manifest.json"
    _require(manifest_path, errors)
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("last_completed") != chapter:
            errors.append("manifest_last_completed_mismatch")
        current_volume = ((chapter - 1) // 40) + 1
        current_volume_node = node_lookup.get(f"L2_Volume{current_volume}", {})
        current_volume_metadata = current_volume_node.get("metadata", {})
        volume_prefix = f"volume_{current_volume:03d}"
        expected_manifest_volume = {
            f"{volume_prefix}_status": current_volume_node.get("status"),
            f"{volume_prefix}_released_chapters": current_volume_metadata.get(
                "released_chapters"
            ),
            f"{volume_prefix}_completed_chapter": current_volume_metadata.get(
                "completed_chapter"
            ),
        }
        for field, expected in expected_manifest_volume.items():
            if manifest.get(field) != expected:
                errors.append(f"manifest_{field}_mismatch")

    return {
        "ok": not errors,
        "chapter": chapter,
        "repaired_index": did_repair,
        "actual": actual,
        **totals,
        "applied_deltas": len(expected_deltas),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one committed M5 TaskGraph chapter.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--chapter", type=int, required=True)
    parser.add_argument("--repair-index", action="store_true")
    args = parser.parse_args()
    result = guard_commit(args.run_root.resolve(), args.chapter, args.repair_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
