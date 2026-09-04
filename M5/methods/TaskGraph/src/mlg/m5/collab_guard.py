from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from mlg.m5.collab_state import (
    committed_memory_closure_errors,
    sha_file,
    sha_payload,
    sha_persisted_payload,
)
from mlg.m5.dataset import ChapterPrompt, han_char_count
from mlg.m5.io import read_json
from mlg.m5.length_policy import (
    CURRENT_LENGTH_POLICY_REVISION,
    LENGTH_BAND_POLICY_REVISION,
    PREVIOUS_LENGTH_POLICY_REVISION,
    chapter_length_policy_payload,
    ranges_for_protocol,
)
from mlg.m5.taskgraph_extract import (
    LIFECYCLE_KEY_SUGGESTION_LIMIT,
    LIFECYCLE_KEY_SUGGESTION_MIN_RATIO,
    MAX_EXTRACTED_L4_FACTS,
    atomic_facts,
)


_REQUIRED_ARTIFACTS = {
    "prewrite_request", "prewrite_response", "prewrite_graph", "writer_input",
    "controller_audit",
    "chapter_text", "writeback_request", "writeback", "final_graph",
}

_WRITEBACK_POLICY_V1 = {
    "schema": "m5-l4-writeback-policy-v1",
    "min_atomic_facts": 1,
    "max_atomic_facts": MAX_EXTRACTED_L4_FACTS,
    "chapter_summary_l4": 1,
    "source_quote": "exact_current_chapter_substring",
    "retry_feedback": "high_confidence_current_chapter_locator-v2",
    "candidate_min_similarity": 0.86,
    "candidate_role": "diagnostic_only",
    "auto_repair": False,
}

_WRITEBACK_POLICY_V2 = {
    "schema": "m5-l4-writeback-policy-v2",
    "min_atomic_facts": 1,
    "max_atomic_facts": MAX_EXTRACTED_L4_FACTS,
    "chapter_summary_l4": 1,
    "source_quote": "model_grounded_nonempty",
    "exact_match": "audit_only",
    "retry_feedback": "missing_source_evidence_only-v1",
    "auto_repair": False,
}

_WRITEBACK_POLICY_V3 = {
    "schema": "m5-l4-writeback-policy-v3",
    "min_atomic_facts": 1,
    "max_atomic_facts": MAX_EXTRACTED_L4_FACTS,
    "chapter_summary_l4": 1,
    "source_quote": "model_grounded_nonempty",
    "exact_match": "audit_only",
    "retry_feedback": "typed_lifecycle_and_source_evidence-v2",
    "lifecycle_key_suggestions": {
        "method": "difflib_sequence_ratio-v1",
        "min_ratio": LIFECYCLE_KEY_SUGGESTION_MIN_RATIO,
        "limit": LIFECYCLE_KEY_SUGGESTION_LIMIT,
        "role": "diagnostic_only",
    },
    "auto_repair": False,
}

_RELAXED_LENGTH_REVISIONS = {
    LENGTH_BAND_POLICY_REVISION,
    PREVIOUS_LENGTH_POLICY_REVISION,
    CURRENT_LENGTH_POLICY_REVISION,
}
_ACTIVITY_POOL_REVISIONS = {
    "prewrite-funnel-activity-decay-v10",
    "prewrite-three-channel-union-v11",
    *_RELAXED_LENGTH_REVISIONS,
}
_THREE_CHANNEL_REVISIONS = {
    "prewrite-three-channel-union-v11",
    *_RELAXED_LENGTH_REVISIONS,
}
_SIX_ATTEMPT_REVISIONS = {
    "prewrite-funnel-deterministic-lifecycle-v9",
    *_ACTIVITY_POOL_REVISIONS,
}
_RELAXED_QUOTE_REVISIONS = {
    "prewrite-funnel-deterministic-lifecycle-v8",
    "prewrite-funnel-deterministic-lifecycle-v9",
    *_ACTIVITY_POOL_REVISIONS,
}


def guard_collaboration_commit(run_dir: Path, chapter_id: int) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    errors: list[str] = []
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        return {"ok": False, "chapter_id": chapter_id, "errors": ["missing_state"]}
    state = read_json(state_path)
    errors.extend(committed_memory_closure_errors(run_dir, state))
    if state.get("schema") != "m5-collaboration-taskgraph-v5":
        errors.append("state_schema")
    state_config = state.get("config")
    revision = (
        state_config.get("protocol_revision")
        if isinstance(state_config, dict) else None
    )
    requested_range, accepted_range = ranges_for_protocol(revision)
    if sha_payload(state_config) != state.get("config_fingerprint"):
        errors.append("state_config_fingerprint")
    if isinstance(state_config, dict):
        if revision not in {
            "prewrite-funnel-deterministic-lifecycle-v5",
            "prewrite-funnel-deterministic-lifecycle-v6",
            "prewrite-funnel-deterministic-lifecycle-v7",
            "prewrite-funnel-deterministic-lifecycle-v8",
            "prewrite-funnel-deterministic-lifecycle-v9",
            "prewrite-funnel-activity-decay-v10",
            "prewrite-three-channel-union-v11",
            LENGTH_BAND_POLICY_REVISION,
            PREVIOUS_LENGTH_POLICY_REVISION,
            CURRENT_LENGTH_POLICY_REVISION,
        }:
            errors.append("state_protocol_revision")
        if revision in {
            "prewrite-funnel-deterministic-lifecycle-v6",
            "prewrite-funnel-deterministic-lifecycle-v7",
        } and state_config.get("writeback_policy") != _WRITEBACK_POLICY_V1:
            errors.append("state_writeback_policy")
        if (
            revision in {
                "prewrite-funnel-deterministic-lifecycle-v8",
                "prewrite-funnel-deterministic-lifecycle-v9",
                "prewrite-funnel-activity-decay-v10",
                "prewrite-three-channel-union-v11",
                LENGTH_BAND_POLICY_REVISION,
            }
            and state_config.get("writeback_policy") != _WRITEBACK_POLICY_V2
        ):
            errors.append("state_writeback_policy")
        if (
            revision in {
                PREVIOUS_LENGTH_POLICY_REVISION,
                CURRENT_LENGTH_POLICY_REVISION,
            }
            and state_config.get("writeback_policy") != _WRITEBACK_POLICY_V3
        ):
            errors.append("state_writeback_policy")
        if (
            revision in _RELAXED_LENGTH_REVISIONS
            and state_config.get("chapter_length_policy")
            != chapter_length_policy_payload()
        ):
            errors.append("state_chapter_length_policy")
    stage = state.get("stage")
    if stage == "pending_commit" and int(state.get("active_chapter") or -1) == chapter_id:
        commit_meta = state.get("transaction", {}).get("artifacts", {})
    elif int(state.get("last_completed", 0)) >= chapter_id:
        commit_meta = state.get("commit_ledger", {}).get(str(chapter_id), {})
    else:
        commit_meta = {}
        errors.append("state_not_pending_or_committed")

    stem = f"chapter_{chapter_id:03d}"
    record_path = run_dir / "records" / f"{stem}.json"
    checkpoint_path = run_dir / "checkpoints" / f"{stem}.json"
    if not record_path.is_file():
        errors.append("missing_record")
    if not checkpoint_path.is_file():
        errors.append("missing_checkpoint")
    if errors and (not record_path.is_file() or not checkpoint_path.is_file()):
        return {"ok": False, "chapter_id": chapter_id, "errors": sorted(set(errors))}

    record = read_json(record_path)
    receipt = read_json(checkpoint_path)
    if record.get("schema") != "m5-collaboration-chapter-record-v5":
        errors.append("record_schema")
    if receipt.get("schema") != "m5-collaboration-commit-receipt-v1":
        errors.append("checkpoint_schema")
    for field in (
        "chapter_id", "transaction_id", "run_mode", "dataset_fingerprint",
        "config_fingerprint", "chapter_prompt_sha256",
    ):
        expected = chapter_id if field == "chapter_id" else record.get(field)
        if receipt.get(field) != expected:
            errors.append(f"checkpoint_{field}_mismatch")
    for field in ("run_mode", "dataset_fingerprint"):
        if record.get(field) != state.get(field):
            errors.append(f"record_{field}_mismatch")
    _validate_record_config_history(state, record, chapter_id, errors)

    artifacts = record.get("artifacts", {})
    if not isinstance(artifacts, dict) or not _REQUIRED_ARTIFACTS <= set(artifacts):
        errors.append("record_artifacts_incomplete")
        artifacts = artifacts if isinstance(artifacts, dict) else {}
    resolved: dict[str, Path] = {}
    for label, descriptor in artifacts.items():
        path = _artifact_path(run_dir, descriptor, errors, label)
        if path is not None:
            resolved[label] = path
    for label in _REQUIRED_ARTIFACTS:
        if label not in resolved:
            errors.append(f"missing_{label}")
    for field, label in (
        ("controller_audit", "controller_audit"),
        ("prewrite_graph", "prewrite_graph"),
        ("writer_input", "writer_input"),
        ("prewrite_response", "prewrite_response"),
        ("final_graph", "final_graph"),
    ):
        descriptor = artifacts.get(label, {})
        if not isinstance(descriptor, dict) or record.get(field) != descriptor.get("path"):
            errors.append(f"record_{field}_path_mismatch")
    receipt_artifacts = receipt.get("artifacts", {})
    for label, descriptor in artifacts.items():
        if receipt_artifacts.get(label) != descriptor:
            errors.append(f"checkpoint_{label}_mismatch")
    _check_commit_artifact(run_dir, commit_meta, "record", record_path, errors)
    _check_commit_artifact(run_dir, commit_meta, "checkpoint", checkpoint_path, errors)
    _check_commit_artifact(
        run_dir, commit_meta, "final_graph", resolved.get("final_graph"), errors,
    )

    chapter_text = ""
    if "chapter_text" in resolved:
        chapter_text = resolved["chapter_text"].read_text(encoding="utf-8-sig")
        if not accepted_range[0] <= han_char_count(chapter_text) <= accepted_range[1]:
            errors.append("chapter_length")
    if stage == "pending_commit":
        if receipt.get("final_memory_sha256") != sha_persisted_payload(
            state.get("memory")
        ):
            errors.append("pending_memory_mismatch")

    pre_graph = _read_if(resolved.get("prewrite_graph"), errors, "prewrite_graph")
    graph = _read_if(resolved.get("final_graph"), errors, "final_graph")
    packet = _read_if(resolved.get("writer_input"), errors, "writer_input")
    controller_audit = _read_if(
        resolved.get("controller_audit"), errors, "controller_audit",
    )
    if pre_graph and graph:
        _validate_graphs(pre_graph, graph, record, errors)
    if packet:
        _validate_writer_packet(
            packet, chapter_id, errors, requested_range=requested_range,
        )
    if controller_audit:
        _validate_controller_audit(
            controller_audit, record, errors,
            require_activity_pool=(
                isinstance(state_config, dict)
                and state_config.get("protocol_revision")
                in _ACTIVITY_POOL_REVISIONS
            ),
            require_three_channel_union=(
                isinstance(state_config, dict)
                and state_config.get("protocol_revision")
                in _THREE_CHANNEL_REVISIONS
            ),
        )
    if record.get("run_mode") == "formal":
        _validate_formal_actor_provenance(
            run_dir, state, record, resolved, errors,
        )
    for label in ("prewrite_request",):
        request = _read_if(resolved.get(label), errors, label)
        if request:
            if request.get("transaction_id") != record.get("transaction_id"):
                errors.append(f"{label}_transaction")
            if request.get("chapter_prompt_sha256") != record.get("chapter_prompt_sha256"):
                errors.append(f"{label}_prompt_hash")
            if (
                isinstance(state_config, dict)
                and state_config.get("protocol_revision")
                in _THREE_CHANNEL_REVISIONS
                and request.get("funnel_formula")
                != "Cfinal=rank_cap(Vc_union_Vr_union_Cdep)"
            ):
                errors.append(f"{label}_funnel_formula")
    request_descriptor = artifacts.get("writeback_request", {})
    writeback_request = _read_if(
        resolved.get("writeback_request"), errors, "writeback_request",
    )
    request_schema = writeback_request.get("schema")
    if request_schema not in {
        "m5-l4-writeback-request-v2", "m5-l4-writeback-request-v3",
    }:
        errors.append("writeback_request_schema")
    if request_schema == "m5-l4-writeback-request-v3":
        audit = writeback_request.get("registry_audit", {})
        registry_record = record.get("writeback_registry", {})
        integer_fields = (
            "total_count", "selected_count", "dropped_count", "prompt_tokens",
            "token_budget", "item_limit",
        )
        if not isinstance(audit, dict) or any(
            not isinstance(audit.get(field), int) for field in integer_fields
        ):
            errors.append("writeback_registry_audit")
        else:
            if (
                audit["selected_count"] + audit["dropped_count"] != audit["total_count"]
                or audit["prompt_tokens"] > audit["token_budget"]
                or audit["selected_count"] > audit["item_limit"]
                or writeback_request.get("full_registry_count") != audit["total_count"]
            ):
                errors.append("writeback_registry_bounds")
            if any(registry_record.get(field) != audit[field] for field in integer_fields):
                errors.append("writeback_registry_record")
        for field in ("full_registry_sha256", "prompt_registry_sha256"):
            value = writeback_request.get(field)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"writeback_registry_{field}")
            if registry_record.get(field) != value:
                errors.append("writeback_registry_record")
        if pre_graph:
            reconstructed_registry = _registry_from_graph(pre_graph)
            if (
                len(reconstructed_registry) != writeback_request.get("full_registry_count")
                or sha_payload(reconstructed_registry)
                != writeback_request.get("full_registry_sha256")
            ):
                errors.append("writeback_full_registry_binding")
        if (
            registry_record.get("controller_source_graph") != record.get("prewrite_graph")
            or registry_record.get("controller_source_graph_sha256")
            != record.get("prewrite_graph_sha256")
        ):
            errors.append("writeback_registry_source_graph")
    if writeback_request.get("chapter_id") != chapter_id:
        errors.append("writeback_request_chapter")
    request_hash = (
        request_descriptor.get("sha256")
        if isinstance(request_descriptor, dict) else None
    )
    if record.get("writeback_request_sha256") != request_hash:
        errors.append("record_writeback_request_hash")
    if record.get("run_mode") == "formal":
        stored_writeback = _read_if(resolved.get("writeback"), errors, "writeback")
        if stored_writeback.get("transaction_id") != record.get("transaction_id"):
            errors.append("writeback_transaction")
        if stored_writeback.get("request_sha256") != request_hash:
            errors.append("writeback_request_hash")
        if not isinstance(stored_writeback.get("response"), dict):
            errors.append("writeback_response_wrapper")
        elif pre_graph and graph and chapter_text:
            _validate_writeback_semantics(
                chapter_id, chapter_text, stored_writeback["response"],
                pre_graph, graph, record, errors,
                require_relaxed_quote_audit=(
                    isinstance(state_config, dict)
                    and state_config.get("protocol_revision")
                    in _RELAXED_QUOTE_REVISIONS
                ),
            )

    try:
        times = [datetime.fromisoformat(record[field]) for field in (
            "candidate_frozen_at", "writer_released_at", "chapter_submitted_at",
            "writeback_submitted_at", "completed_at",
        )]
        if times != sorted(times):
            errors.append("transaction_time_order")
    except (KeyError, TypeError, ValueError):
        errors.append("transaction_timestamps")
    return {
        "ok": not errors,
        "chapter_id": chapter_id,
        "selected_prewrite_nodes": len(record.get("selected_node_ids", [])),
        "errors": sorted(set(errors)),
    }


def _validate_writeback_semantics(
    chapter_id: int,
    chapter_text: str,
    response: dict[str, Any],
    pre_graph: dict[str, Any],
    graph: dict[str, Any],
    record: dict[str, Any],
    errors: list[str],
    *,
    require_relaxed_quote_audit: bool = False,
) -> None:
    prompt = ChapterPrompt(
        chapter_id, 1, "", "", (), (), 2500,
    )
    registered_keys = {row["key"] for row in _registry_from_graph(pre_graph)}
    try:
        parsed = atomic_facts(
            prompt, response, source_text=chapter_text,
            registered_keys=registered_keys, strict=True,
        )
    except (TypeError, ValueError):
        errors.append("writeback_semantic_validation")
        return
    fact_count = len(parsed)
    if not 1 <= fact_count <= MAX_EXTRACTED_L4_FACTS:
        errors.append("writeback_atomic_fact_count")
    validation = record.get("writeback_validation", {})
    if not isinstance(validation, dict):
        errors.append("record_writeback_validation")
    else:
        if validation.get("facts_accepted") != fact_count:
            errors.append("record_writeback_fact_count")
        if validation.get("source_quotes_valid") != fact_count:
            errors.append("record_writeback_source_quote_count")
        quote_policy = validation.get("source_quote_policy")
        if (
            (require_relaxed_quote_audit or quote_policy is not None)
            and quote_policy != "model-grounded-nonempty-v1"
        ):
            errors.append("record_writeback_source_quote_policy")
        if quote_policy == "model-grounded-nonempty-v1":
            nonexact = [
                fact["key"] for fact in parsed
                if fact.get("source_quote_exact") is False
            ]
            if validation.get("source_quotes_exact") != fact_count - len(nonexact):
                errors.append("record_writeback_source_quote_exact_count")
            if record.get("nonexact_writeback_source_quotes") != nonexact:
                errors.append("record_writeback_nonexact_source_quotes")
        for operation in ("create", "update"):
            expected = sum(fact["operation"] == operation for fact in parsed)
            if validation.get(operation) != expected:
                errors.append(f"record_writeback_{operation}_count")
        if validation.get("create", 0) + validation.get("update", 0) != fact_count:
            errors.append("record_writeback_operation_total")
    current_l4 = [
        node for node in graph.get("nodes", [])
        if isinstance(node, dict)
        and node.get("level") == "L4"
        and int(node.get("metadata", {}).get("chapter_id", 0) or 0) == chapter_id
    ]
    summaries = [
        node for node in current_l4
        if str(node.get("metadata", {}).get("state_key", ""))
        == f"chapter_summary:{chapter_id}"
    ]
    if len(current_l4) != fact_count + 1:
        errors.append("graph_current_l4_count")
    if len(summaries) != 1:
        errors.append("graph_current_summary_l4_count")


def _validate_formal_actor_provenance(
    run_dir: Path, state: dict[str, Any], record: dict[str, Any],
    resolved: dict[str, Path], errors: list[str],
) -> None:
    actors = record.get("actor_provenance")
    roles = ("prewrite_judge", "writer", "writeback_extractor")
    if not isinstance(actors, dict) or set(actors) != set(roles):
        errors.append("formal_actor_roles")
        return
    state_config = state.get("config")
    raw_policy = (
        state_config.get("controller_policy", {})
        if isinstance(state_config, dict) else {}
    )
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    role_policies = policy.get("roles", {})
    backend = policy.get("backend")
    policy_schema = policy.get("schema")
    if not isinstance(raw_policy, dict) or policy_schema not in {
        "m5-taskgraph-controller-policy-v1",
        "m5-taskgraph-controller-policy-v2",
    }:
        errors.append("formal_controller_policy_schema")
    if backend != "codex_cli_ephemeral":
        errors.append("formal_controller_policy_backend")
    expected_isolation = (
        "windows_elevated_acl_permission_profile_v1"
        if policy_schema == "m5-taskgraph-controller-policy-v2"
        else "fresh_ephemeral_read_only"
    )
    if policy.get("isolation") != expected_isolation:
        errors.append("formal_controller_policy_isolation")
    evidence_hash = policy.get("isolation_evidence_sha256")
    if policy_schema == "m5-taskgraph-controller-policy-v2":
        evidence = policy.get("isolation_evidence")
        if (
            not isinstance(evidence, dict)
            or evidence.get("probe_passed") is not True
            or evidence.get("implementation") != (
                "windows_elevated_acl_and_permission_profile"
            )
            or not isinstance(evidence.get("acl_probe_count"), int)
            or evidence.get("acl_probe_count") != evidence.get("forbidden_probe_count")
            or evidence.get("acl_probe_count", 0) < 1
            or sha_payload(evidence) != evidence_hash
        ):
            errors.append("formal_controller_policy_isolation_evidence")
    max_attempts = policy.get("max_attempts")
    revision = (
        state_config.get("protocol_revision")
        if isinstance(state_config, dict) else None
    )
    if (
        not isinstance(max_attempts, int) or isinstance(max_attempts, bool)
        or max_attempts < 1
        or (
            revision in _SIX_ATTEMPT_REVISIONS
            and max_attempts != 6
        )
    ):
        errors.append("formal_controller_policy_max_attempts")
        max_attempts = 0
    artifacts = record.get("artifacts", {})
    request_hashes: dict[str, Any] = {}
    for role, label in {
        "prewrite_judge": "prewrite_request", "writer": "writer_input",
        "writeback_extractor": "writeback_request",
    }.items():
        descriptor = artifacts.get(label) if isinstance(artifacts, dict) else None
        request_hashes[role] = (
            descriptor.get("sha256") if isinstance(descriptor, dict) else None
        )
    accepted_outputs: dict[str, Path] = {}
    global_ids: set[str] = set()
    for role in roles:
        actor = actors.get(role)
        if not isinstance(actor, dict):
            errors.append(f"formal_actor_{role}")
            continue
        for field in ("model", "reasoning_effort", "agent_id"):
            if not actor.get(field):
                errors.append(f"formal_actor_{role}_{field}")
        role_policy = role_policies.get(role, {}) if isinstance(role_policies, dict) else {}
        if not isinstance(role_policy, dict):
            role_policy = {}
        if actor.get("model") != role_policy.get("model"):
            errors.append(f"formal_actor_{role}_policy_model")
        if actor.get("reasoning_effort") != role_policy.get("reasoning_effort"):
            errors.append(f"formal_actor_{role}_policy_effort")
        invocations = actor.get("invocations")
        attempts = actor.get("attempt_artifacts")
        if not isinstance(invocations, list) or not invocations:
            errors.append(f"formal_actor_{role}_invocations")
            continue
        if not isinstance(attempts, list) or not attempts:
            errors.append(f"formal_actor_{role}_attempt_artifacts")
            continue
        invocation_ids: set[str] = set()
        invocation_by_id: dict[str, dict[str, Any]] = {}
        for invocation in invocations:
            if not isinstance(invocation, dict):
                errors.append(f"formal_actor_{role}_invocation_row")
                continue
            invocation_id = str(invocation.get("invocation_id") or "")
            if not invocation_id or invocation_id in invocation_ids or invocation_id in global_ids:
                errors.append("formal_actor_invocation_id_unique")
            invocation_ids.add(invocation_id)
            global_ids.add(invocation_id)
            invocation_by_id[invocation_id] = invocation
            if invocation.get("model") != role_policy.get("model"):
                errors.append(f"formal_actor_{role}_invocation_model")
            if invocation.get("reasoning_effort") != role_policy.get("reasoning_effort"):
                errors.append(f"formal_actor_{role}_invocation_effort")
            if invocation.get("backend") != backend:
                errors.append(f"formal_actor_{role}_invocation_backend")
        attempt_ids: set[str] = set()
        attempt_numbers: set[int] = set()
        accepted = 0
        for attempt in attempts:
            if not isinstance(attempt, dict):
                errors.append(f"formal_actor_{role}_attempt_row")
                continue
            number = attempt.get("attempt")
            if not isinstance(number, int) or isinstance(number, bool) or number < 1:
                errors.append(f"formal_actor_{role}_attempt_number")
            elif number in attempt_numbers:
                errors.append(f"formal_actor_{role}_attempt_duplicate")
            else:
                attempt_numbers.add(number)
            invocation_id = str(attempt.get("invocation_id") or "")
            if not invocation_id or invocation_id in attempt_ids:
                errors.append(f"formal_actor_{role}_attempt_invocation")
            attempt_ids.add(invocation_id)
            is_accepted = attempt.get("accepted") is True
            accepted += int(is_accepted)
            audit_path = _artifact_path(
                run_dir, attempt.get("audit"), errors,
                f"actor_{role}_attempt_{number}_audit",
            )
            output = attempt.get("output")
            output_path = _artifact_path(
                run_dir, output, errors,
                f"actor_{role}_attempt_{number}_output",
            ) if output is not None else None
            if is_accepted and output_path is None:
                errors.append(f"formal_actor_{role}_accepted_output")
            elif is_accepted and output_path is not None:
                accepted_outputs[role] = output_path
            if audit_path is not None:
                audit = _read_if(audit_path, errors, f"actor_{role}_attempt_audit")
                call = audit.get("call", {}) if isinstance(audit, dict) else {}
                if audit.get("schema") != "m5-codex-cli-attempt-v1":
                    errors.append(f"formal_actor_{role}_attempt_schema")
                if audit.get("audit_status") != "finalized":
                    errors.append(f"formal_actor_{role}_attempt_not_finalized")
                if audit.get("chapter_id") != record.get("chapter_id"):
                    errors.append(f"formal_actor_{role}_attempt_chapter")
                if audit.get("stage") != role:
                    errors.append(f"formal_actor_{role}_attempt_stage")
                if audit.get("request_sha256") != request_hashes.get(role):
                    errors.append(f"formal_actor_{role}_attempt_request")
                if audit.get("accepted") is not is_accepted:
                    errors.append(f"formal_actor_{role}_attempt_acceptance")
                if not isinstance(call, dict) or call.get("invocation_id") != invocation_id:
                    errors.append(f"formal_actor_{role}_attempt_call")
                elif call != invocation_by_id.get(invocation_id):
                    errors.append(f"formal_actor_{role}_attempt_call_record")
                if audit.get("transaction_id") != record.get("transaction_id"):
                    errors.append(f"formal_actor_{role}_attempt_transaction")
                output_descriptor = output if isinstance(output, dict) else {}
                if output_descriptor:
                    if audit.get("output_path") != output_descriptor.get("path"):
                        errors.append(f"formal_actor_{role}_attempt_output_path")
                    if audit.get("output_sha256") != output_descriptor.get("sha256"):
                        errors.append(f"formal_actor_{role}_attempt_output_hash")
                elif audit.get("output_sha256") is not None:
                    errors.append(f"formal_actor_{role}_attempt_output_hash")
                raw_output_path = audit.get("output_path")
                try:
                    if not isinstance(raw_output_path, str) or not raw_output_path:
                        raise ValueError
                    (run_dir / raw_output_path).resolve().relative_to(run_dir)
                except ValueError:
                    errors.append(f"formal_actor_{role}_attempt_output_path")
                status = call.get("status") if isinstance(call, dict) else None
                if status not in {"succeeded", "failed", "timeout"}:
                    errors.append(f"formal_actor_{role}_attempt_call_status")
                if is_accepted and status != "succeeded":
                    errors.append(f"formal_actor_{role}_accepted_call_status")
                if call.get("attempt") != number:
                    errors.append(f"formal_actor_{role}_attempt_call_number")
                expected_purpose = {
                    "prewrite_judge": "prewrite_judge", "writer": "chapter",
                    "writeback_extractor": "writeback_extractor",
                }[role]
                if call.get("purpose") != expected_purpose:
                    errors.append(f"formal_actor_{role}_attempt_purpose")
                if call.get("session_reuse") is not False:
                    errors.append(f"formal_actor_{role}_attempt_session_reuse")
                if call.get("backend") != backend:
                    errors.append(f"formal_actor_{role}_attempt_backend")
                if call.get("isolation") != policy.get("isolation"):
                    errors.append(f"formal_actor_{role}_attempt_isolation")
                if policy_schema == "m5-taskgraph-controller-policy-v2" and (
                    call.get("isolation_evidence_sha256") != evidence_hash
                ):
                    errors.append(f"formal_actor_{role}_attempt_isolation_evidence")
        if accepted != 1:
            errors.append(f"formal_actor_{role}_accepted_count")
        if attempt_numbers != set(range(1, len(attempt_numbers) + 1)):
            errors.append(f"formal_actor_{role}_attempt_sequence")
        if attempt_numbers and max(attempt_numbers) > max_attempts:
            errors.append(f"formal_actor_{role}_attempt_limit")
        accepted_ids = {
            str(row.get("invocation_id")) for row in attempts
            if isinstance(row, dict) and row.get("accepted") is True
        }
        if accepted_ids != {str(actor.get("agent_id"))}:
            errors.append(f"formal_actor_{role}_agent_id")
        if attempt_ids != invocation_ids:
            errors.append(f"formal_actor_{role}_attempt_invocation_set")
        sums = {
            "input_tokens": sum(int(row.get("input_tokens") or 0) for row in invocations),
            "output_tokens": sum(int(row.get("output_tokens") or 0) for row in invocations),
            "elapsed_ms": sum(float(row.get("elapsed_ms") or 0) for row in invocations),
        }
        for field, expected in sums.items():
            try:
                actual = float(actor.get(field))
            except (TypeError, ValueError):
                errors.append(f"formal_actor_{role}_{field}_aggregate")
                continue
            if abs(actual - float(expected)) > 1e-6:
                errors.append(f"formal_actor_{role}_{field}_aggregate")
    _validate_accepted_actor_outputs(
        accepted_outputs, resolved, record, errors,
    )


def _validate_accepted_actor_outputs(
    accepted: dict[str, Path], resolved: dict[str, Path],
    record: dict[str, Any], errors: list[str],
) -> None:
    prewrite = _read_if(
        resolved.get("prewrite_response"), errors, "prewrite_response",
    )
    prewrite_output = _read_if(
        accepted.get("prewrite_judge"), errors, "accepted_prewrite_output",
    )
    if prewrite_output != prewrite.get("response"):
        errors.append("accepted_prewrite_output_mismatch")

    writer_path = accepted.get("writer")
    chapter_path = resolved.get("chapter_text")
    if writer_path is None or chapter_path is None:
        errors.append("accepted_writer_output_missing")
    else:
        try:
            writer_text = writer_path.read_text(encoding="utf-8-sig").strip()
            chapter_text = chapter_path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            errors.append("accepted_writer_output_read")
        else:
            if writer_text != chapter_text:
                errors.append("accepted_writer_output_mismatch")

    writeback = _read_if(resolved.get("writeback"), errors, "writeback")
    writeback_output = _read_if(
        accepted.get("writeback_extractor"), errors,
        "accepted_writeback_output",
    )
    if writeback_output != writeback.get("response"):
        errors.append("accepted_writeback_output_mismatch")
    if writeback.get("transaction_id") != record.get("transaction_id"):
        errors.append("accepted_writeback_transaction")


def _validate_record_config_history(
    state: dict[str, Any], record: dict[str, Any], chapter_id: int,
    errors: list[str],
) -> None:
    """Allow immutable pre-migration records only inside their declared range."""
    record_fingerprint = record.get("config_fingerprint")
    if record_fingerprint == state.get("config_fingerprint"):
        return
    history = state.get("config_history", {})
    row = history.get(record_fingerprint) if isinstance(history, dict) else None
    if not isinstance(row, dict):
        errors.append("record_config_fingerprint_mismatch")
        return
    historical_config = row.get("config")
    if (
        not isinstance(historical_config, dict)
        or sha_payload(historical_config) != record_fingerprint
    ):
        errors.append("state_config_history_fingerprint")
    chapter_range = row.get("chapter_range")
    if (
        not isinstance(chapter_range, list)
        or len(chapter_range) != 2
        or not all(isinstance(value, int) for value in chapter_range)
        or not chapter_range[0] <= chapter_id <= chapter_range[1]
    ):
        errors.append("record_config_history_range")


def _artifact_path(
    run_dir: Path, descriptor: Any, errors: list[str], label: str,
) -> Path | None:
    if not isinstance(descriptor, dict):
        errors.append(f"invalid_{label}_descriptor")
        return None
    try:
        path = (run_dir / str(descriptor["path"])).resolve()
        path.relative_to(run_dir)
    except (KeyError, ValueError):
        errors.append(f"invalid_{label}_path")
        return None
    if not path.is_file():
        errors.append(f"missing_{label}")
        return None
    if descriptor.get("sha256") != sha_file(path):
        errors.append(f"{label}_sha256_mismatch")
    return path


def _check_commit_artifact(
    run_dir: Path, meta: Any, label: str, expected: Path | None, errors: list[str],
) -> None:
    descriptor = meta.get(label) if isinstance(meta, dict) else None
    path = _artifact_path(run_dir, descriptor, errors, f"commit_{label}")
    if expected is not None and path != expected:
        errors.append(f"commit_{label}_path_mismatch")


def _read_if(path: Path | None, errors: list[str], label: str) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = read_json(path)
    except (OSError, ValueError):
        errors.append(f"invalid_{label}_json")
        return {}
    if not isinstance(value, dict):
        errors.append(f"invalid_{label}_json")
        return {}
    return value


def _validate_writer_packet(
    packet: dict[str, Any], chapter_id: int, errors: list[str],
    *, requested_range: tuple[int, int] = (2000, 3000),
) -> None:
    if set(packet) != {"schema", "system", "user", "generation"}:
        errors.append("writer_input_private_metadata")
    if packet.get("schema") != "m5-writer-packet-v2":
        errors.append("writer_input_schema")
    generation = packet.get("generation", {})
    allowed_generation = {
        "chapter_id", "target_han_chars", "allowed_han_chars", "output_format",
    }
    if not isinstance(generation, dict) or set(generation) != allowed_generation:
        errors.append("writer_input_generation_metadata")
    if not isinstance(generation, dict) or generation.get("chapter_id") != chapter_id:
        errors.append("writer_input_chapter")
    if isinstance(generation, dict) and (
        generation.get("target_han_chars") != 2500
        or generation.get("allowed_han_chars") != list(requested_range)
    ):
        errors.append("writer_input_length_request")
    visible = f"{packet.get('system', '')}\n{packet.get('user', '')}"
    forbidden = (
        "TaskGraph", "伏笔关联候选", "dependency_type", "stable_key", "node_id",
        "memory_audit", "prewrite_graph", "candidate_channels",
    )
    if any(marker in visible for marker in forbidden) or re.search(
        r"\bL[1-4]_[A-Za-z0-9_]+\b", visible,
    ):
        errors.append("writer_input_internal_metadata")


def _validate_controller_audit(
    audit: dict[str, Any], record: dict[str, Any], errors: list[str],
    *, require_activity_pool: bool = False,
    require_three_channel_union: bool = False,
) -> None:
    if audit.get("schema") != "m5-controller-prewrite-audit-v1":
        errors.append("controller_audit_schema")
    for field in ("chapter_id", "transaction_id", "target_id"):
        if audit.get(field) != record.get(field):
            errors.append(f"controller_audit_{field}")
    if audit.get("prewrite_graph") != record.get("prewrite_graph"):
        errors.append("controller_audit_prewrite_graph")
    memory = audit.get("memory", {})
    if not isinstance(memory, dict):
        errors.append("controller_audit_memory")
        return
    checks = {
        "selected_node_ids": record.get("selected_node_ids"),
        "memory_tokens": record.get("memory_tokens"),
        "memory_blocks": record.get("memory_blocks"),
        "deduplicated_count": record.get("memory_deduplicated_count"),
        "dropped_count": record.get("memory_dropped_count"),
    }
    for field, expected in checks.items():
        if memory.get(field) != expected:
            errors.append(f"controller_audit_{field}")
    prompt = audit.get("writer_prompt", {})
    if not isinstance(prompt, dict) or prompt.get("total_tokens") != record.get(
        "writer_prompt_tokens"
    ):
        errors.append("controller_audit_writer_tokens")
    dependency = audit.get("dependency_audit", {})
    used = {
        str(row.get("source_id"))
        for row in dependency.get("created", [])
        if row.get("used_in_prompt") is True
    } if isinstance(dependency, dict) else set()
    if used != set(record.get("selected_node_ids", [])):
        errors.append("controller_audit_selected_dependencies")
    if require_activity_pool:
        _validate_activity_pool_audit(
            dependency, int(record.get("chapter_id", 0) or 0), errors,
        )
    if require_three_channel_union:
        _validate_three_channel_union_audit(dependency, errors)


def _validate_three_channel_union_audit(
    dependency: dict[str, Any], errors: list[str],
) -> None:
    if not isinstance(dependency, dict):
        errors.append("controller_audit_candidate_union")
        return
    if (
        dependency.get("method") != "paper_vc_cdep_vr_model_judge_v6"
        or dependency.get("fusion_formula")
        != "Cfinal=rank_cap(Vc_union_Vr_union_Cdep)"
    ):
        errors.append("controller_audit_candidate_formula")
    rows: dict[str, list[Any]] = {}
    for name in ("Vc", "Vr", "Cdep", "Cfinal"):
        value = dependency.get(name)
        if (
            not isinstance(value, list)
            or not all(isinstance(item, str) and item for item in value)
            or len(value) != len(set(value))
        ):
            errors.append(f"controller_audit_{name}_set")
            return
        rows[name] = value
    union = set(rows["Vc"]) | set(rows["Vr"]) | set(rows["Cdep"])
    union_count = dependency.get("candidate_union_count")
    if (
        not isinstance(union_count, int) or isinstance(union_count, bool)
        or union_count != len(union)
    ):
        errors.append("controller_audit_candidate_union_count")
    final = rows["Cfinal"]
    config = dependency.get("config", {})
    limit = config.get("candidate_limit") if isinstance(config, dict) else None
    expected_config = {
        "w": 12,
        "theta": 0.35,
        "semantic_top_k": 128,
        "candidate_limit": 192,
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected_config.items()
    ):
        errors.append("controller_audit_candidate_config")
    if (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        or len(final) > limit or not set(final) <= union
        or (len(union) <= limit and set(final) != union)
    ):
        errors.append("controller_audit_candidate_union")
    ranking = dependency.get("candidate_ranking")
    ranking_ids: list[str] = []
    ranking_keys: list[tuple[Any, ...]] = []
    ranking_valid = isinstance(ranking, list)
    if ranking_valid:
        for row in ranking:
            if not isinstance(row, dict):
                ranking_valid = False
                break
            source_id = row.get("source_id")
            insertion = row.get("insertion_index")
            raw_scores = [
                row.get("fused_score"), row.get("semantic_score"),
                row.get("task_relevance"), row.get("temporal_relevance"),
            ]
            if (
                not isinstance(source_id, str) or not source_id
                or not isinstance(insertion, int) or isinstance(insertion, bool)
                or insertion < 0
                or any(isinstance(value, bool) for value in raw_scores)
            ):
                ranking_valid = False
                break
            try:
                scores = [float(value) for value in raw_scores]
            except (TypeError, ValueError):
                ranking_valid = False
                break
            if any(not math.isfinite(value) for value in scores):
                ranking_valid = False
                break
            ranking_ids.append(source_id)
            ranking_keys.append((
                -scores[0], -scores[1], -scores[2], -scores[3],
                insertion, source_id,
            ))
    if (
        not ranking_valid or len(ranking_ids) != len(union)
        or len(ranking_ids) != len(set(ranking_ids))
        or set(ranking_ids) != union
        or ranking_keys != sorted(ranking_keys)
        or not isinstance(limit, int) or isinstance(limit, bool)
        or final != ranking_ids[:limit]
    ):
        errors.append("controller_audit_candidate_ranking")
    details = dependency.get("candidate_details")
    if not isinstance(details, list) or [
        row.get("source_id") for row in details if isinstance(row, dict)
    ] != final or len(details) != len(final):
        errors.append("controller_audit_candidate_details")
        return
    channel_sets = {name: set(rows[name]) for name in ("Vc", "Vr", "Cdep")}
    for row in details:
        source_id = row.get("source_id")
        expected = [
            name for name in ("Vc", "Vr", "Cdep")
            if source_id in channel_sets[name]
        ]
        if row.get("candidate_channels") != expected:
            errors.append("controller_audit_candidate_channels")


def _validate_activity_pool_audit(
    dependency: dict[str, Any], target_chapter: int, errors: list[str],
) -> None:
    pool = dependency.get("activity_pool")
    config = dependency.get("config")
    if not isinstance(pool, dict) or not isinstance(config, dict):
        errors.append("controller_audit_activity_pool")
        return
    expected_config = {
        "active_pool_limit": 100,
        "activity_half_life": 40.0,
        "activity_unresolved_weight": 0.45,
        "activity_frequency_weight": 0.30,
        "activity_importance_weight": 0.25,
        "activity_frequency_saturation": 3.0,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        errors.append("controller_audit_activity_config")
    raw = pool.get("raw_count")
    selected_count = pool.get("selected_count")
    dormant = pool.get("dormant_count")
    selected = pool.get("selected")
    vr = dependency.get("Vr")
    if (
        not all(isinstance(value, int) and not isinstance(value, bool)
                for value in (raw, selected_count, dormant))
        or not isinstance(selected, list)
        or not isinstance(vr, list)
    ):
        errors.append("controller_audit_activity_counts")
        return
    if (
        raw < 0 or selected_count < 0 or dormant < 0
        or selected_count > 100 or selected_count > raw
        or dormant != raw - selected_count
        or len(selected) != selected_count or len(vr) != selected_count
    ):
        errors.append("controller_audit_activity_counts")
    ids = [row.get("source_id") for row in selected if isinstance(row, dict)]
    if len(ids) != len(selected) or ids != vr or len(set(ids)) != len(ids):
        errors.append("controller_audit_activity_members")
    prior_score = float("inf")
    for row in selected:
        if not isinstance(row, dict):
            errors.append("controller_audit_activity_row")
            continue
        age = row.get("age")
        last_touch = row.get("last_touch_chapter")
        ref_count = row.get("ref_count")
        unresolved = row.get("unresolved")
        if (
            not all(isinstance(value, int) and not isinstance(value, bool)
                    for value in (age, last_touch, ref_count))
            or not isinstance(unresolved, bool)
        ):
            errors.append("controller_audit_activity_row")
            continue
        try:
            score = float(row["activity_score"])
            frequency = float(row["frequency"])
            importance = float(row["importance"])
        except (KeyError, TypeError, ValueError):
            errors.append("controller_audit_activity_row")
            continue
        if (
            not math.isfinite(score) or score < 0 or score > prior_score + 1e-6
            or age < 0 or last_touch < 0 or ref_count < 0
            or not 0.0 <= frequency <= 1.0
            or not 0.0 <= importance <= 5.0
            or age != max(0, target_chapter - last_touch)
        ):
            errors.append("controller_audit_activity_row")
        expected_frequency = min(1.0, math.log2(1.0 + ref_count) / 3.0)
        if abs(frequency - round(expected_frequency, 6)) > 2e-6:
            errors.append("controller_audit_activity_frequency")
        expected = (
            0.45 * int(unresolved) + 0.30 * expected_frequency
            + 0.25 * (importance / 5.0)
        ) * (2.0 ** (-age / 40.0))
        if abs(score - round(expected, 6)) > 2e-6:
            errors.append("controller_audit_activity_score")
        prior_score = score


def _validate_graphs(
    pre_graph: dict[str, Any], graph: dict[str, Any], record: dict[str, Any],
    errors: list[str],
) -> None:
    pre_edges = [edge for edge in pre_graph.get("edges", [])]
    final_edges = [edge for edge in graph.get("edges", [])]
    if any(edge not in final_edges for edge in pre_edges):
        errors.append("prewrite_edges_changed_after_generation")
    nodes = {node.get("node_id"): node for node in graph.get("nodes", [])}
    dependencies = [
        edge for edge in final_edges if edge.get("edge_type") == "DEPENDENCY"
    ]
    pre_all_dependencies = [
        edge for edge in pre_edges if edge.get("edge_type") == "DEPENDENCY"
    ]
    target_id = record.get("target_id")
    completed_l3 = nodes.get(target_id)
    completed_metadata = (
        completed_l3.get("metadata", {}) if isinstance(completed_l3, dict) else {}
    )
    if (
        not isinstance(completed_l3, dict)
        or completed_l3.get("level") != "L3"
        or completed_l3.get("status") != "Done"
        or completed_metadata.get("embedding_representation") != "actual_completed_l3"
        or completed_metadata.get("result_source")
        != "committed_chapter_with_extractor_summary"
        or completed_metadata.get("result_text_sha256") != record.get("text_sha256")
        or completed_l3.get("value") != completed_metadata.get("result_summary")
    ):
        errors.append("completed_l3_result_binding")
    pre_target_dependencies = [
        edge for edge in pre_all_dependencies
        if edge.get("edge_type") == "DEPENDENCY" and edge.get("target_id") == target_id
    ]
    used_sources = {
        edge.get("source_id") for edge in pre_target_dependencies
        if edge.get("metadata", {}).get("used_in_prompt") is True
    }
    if used_sources != set(record.get("selected_node_ids", [])):
        errors.append("selected_nodes_do_not_match_prewrite_edges")
    if any(
        edge.get("metadata", {}).get("selection_phase") != "pre_write"
        for edge in pre_target_dependencies
    ):
        errors.append("invalid_prewrite_dependency_phase")
    _validate_dependency_delta(
        pre_all_dependencies, dependencies, record, errors,
    )
    for edge in dependencies:
        source = nodes.get(edge.get("source_id"))
        target = nodes.get(edge.get("target_id"))
        if source is None or target is None:
            errors.append("dependency_missing_node")
            continue
        if source.get("level") == "L1":
            errors.append("dependency_from_l1")
        if int(source.get("turn_index", 0)) > int(target.get("turn_index", 0)):
            errors.append("dependency_from_future_turn")
        source_chapter = int(source.get("metadata", {}).get("chapter_id", 0))
        target_chapter = int(target.get("metadata", {}).get("chapter_id", 0))
        if source_chapter and target_chapter and source_chapter >= target_chapter:
            errors.append("dependency_from_same_or_future_chapter")
        if source.get("level") == "L2" and source.get("status") == "Pending":
            errors.append("dependency_from_pending_volume")
        source_volume = int(source.get("metadata", {}).get("volume_id", 0))
        target_volume = int(target.get("metadata", {}).get("volume_id", 0))
        if source.get("level") == "L2" and target_volume and source_volume > target_volume:
            errors.append("dependency_from_future_volume")
        phase = edge.get("metadata", {}).get("selection_phase")
        if phase == "pre_write":
            if target.get("level") != "L3":
                errors.append("prewrite_dependency_target_not_l3")
        elif phase == "state_lifecycle":
            if source.get("level") != "L4" or target.get("level") != "L4":
                errors.append("lifecycle_dependency_not_l4_to_l4")
            if edge.get("metadata", {}).get("used_in_prompt"):
                errors.append("lifecycle_edge_marked_as_prompt_context")
        else:
            errors.append("invalid_dependency_phase")
    if any(edge.get("edge_type") == "MAINLINE" for edge in final_edges):
        errors.append("mainline_present")
    _validate_lifecycle_edges(nodes, dependencies, record, errors)
    _validate_inclusion(nodes, final_edges, errors)


def _registry_from_graph(graph: dict[str, Any]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        metadata = node.get("metadata", {}) if isinstance(node, dict) else {}
        key = str(metadata.get("state_key", ""))
        status = str(metadata.get("state_status", ""))
        if (
            node.get("level") != "L4" or not key
            or key.startswith("chapter_summary:")
            or status not in {"active", "resolved"}
        ):
            continue
        row = {
            "key": key,
            "value": node.get("value", ""),
            "entities": list(metadata.get("entities", [])),
            "chapter_id": int(metadata.get("chapter_id", 0)),
            "final_status": status,
            "version": int(metadata.get("state_version", 1)),
        }
        prior = latest.get(key)
        if prior is None or (row["version"], row["chapter_id"]) > (
            prior["version"], prior["chapter_id"],
        ):
            latest[key] = row
    return [latest[key] for key in sorted(latest)]


def _validate_dependency_delta(
    pre_dependencies: list[dict[str, Any]], final_dependencies: list[dict[str, Any]],
    record: dict[str, Any], errors: list[str],
) -> None:
    unmatched = list(pre_dependencies)
    added: list[dict[str, Any]] = []
    for edge in final_dependencies:
        try:
            index = unmatched.index(edge)
        except ValueError:
            added.append(edge)
        else:
            unmatched.pop(index)
    declared = [
        row for row in record.get("lifecycle_edges", []) if isinstance(row, dict)
    ]
    added_pairs = [
        (str(edge.get("source_id")), str(edge.get("target_id"))) for edge in added
    ]
    declared_pairs = [
        (str(row.get("source_id")), str(row.get("target_id"))) for row in declared
    ]
    if unmatched or sorted(added_pairs) != sorted(declared_pairs):
        errors.append("unexpected_dependency_delta")
    for edge in added:
        metadata = edge.get("metadata", {})
        if (
            metadata.get("selection_phase") != "state_lifecycle"
            or metadata.get("method") != "stable_key_lifecycle_v1"
            or metadata.get("dependency_type") != "state_continuity"
            or metadata.get("stable_key_match") is not True
        ):
            errors.append("unexpected_dependency_delta")


def _validate_lifecycle_edges(
    nodes: dict[str, dict[str, Any]], dependencies: list[dict[str, Any]],
    record: dict[str, Any], errors: list[str],
) -> None:
    lifecycle_edges = [
        edge for edge in dependencies
        if edge.get("metadata", {}).get("selection_phase") == "state_lifecycle"
    ]
    lifecycle_sources = {
        str(edge.get("source_id")) for edge in lifecycle_edges
    }
    incoming: dict[str, list[dict[str, Any]]] = {}
    for edge in lifecycle_edges:
        source = nodes.get(edge.get("source_id"), {})
        target = nodes.get(edge.get("target_id"), {})
        metadata = edge.get("metadata", {})
        incoming.setdefault(str(edge.get("target_id")), []).append(edge)
        source_meta = source.get("metadata", {})
        target_meta = target.get("metadata", {})
        source_key = source_meta.get("state_key")
        target_key = target_meta.get("state_key")
        source_chapter = int(source_meta.get("chapter_id", 0) or 0)
        target_chapter = int(target_meta.get("chapter_id", 0) or 0)
        if source.get("level") != "L4" or target.get("level") != "L4":
            errors.append("lifecycle_edge_level")
        if source_chapter >= target_chapter:
            errors.append("lifecycle_edge_not_cross_chapter")
        if not source_key or source_key != target_key:
            errors.append("lifecycle_edge_key_mismatch")
        if metadata.get("dependency_type") != "state_continuity":
            errors.append("lifecycle_edge_type")
        if metadata.get("method") != "stable_key_lifecycle_v1":
            errors.append("lifecycle_edge_method")
        if metadata.get("stable_key_match") is not True:
            errors.append("lifecycle_edge_not_stable")
        if metadata.get("lifecycle_operation") != "update":
            errors.append("lifecycle_edge_operation")
        if metadata.get("state_key") != target_key:
            errors.append("lifecycle_edge_metadata_key")
        if source_meta.get("state_status") != "superseded":
            errors.append("lifecycle_source_not_superseded")
        if source_meta.get("superseded_by") != target.get("node_id"):
            errors.append("lifecycle_superseded_pointer")
        if source_meta.get("superseded_at_chapter") != target_chapter:
            errors.append("lifecycle_superseded_chapter")
        if target_meta.get("lifecycle_operation") != "update":
            errors.append("lifecycle_target_not_update")
        if target_meta.get("prior_key") != target_key:
            errors.append("lifecycle_target_prior_key")
        if target_meta.get("prior_node_id") != source.get("node_id"):
            errors.append("lifecycle_target_prior_node")
        target_id = str(target.get("node_id"))
        target_status = target_meta.get("state_status")
        if target_id in lifecycle_sources and target_status != "superseded":
            errors.append("lifecycle_intermediate_target_status")
        if (
            target_id not in lifecycle_sources
            and target_status not in {"active", "resolved"}
        ):
            errors.append("lifecycle_target_final_status")
        if int(target_meta.get("state_version", 0) or 0) != int(
            source_meta.get("state_version", 0) or 0
        ) + 1:
            errors.append("lifecycle_state_version")

    for node_id, node in nodes.items():
        if node.get("level") != "L4":
            continue
        operation = node.get("metadata", {}).get("lifecycle_operation")
        count = len(incoming.get(node_id, []))
        if operation == "update" and count != 1:
            errors.append("lifecycle_update_parent_count")
        if operation == "create" and count:
            errors.append("lifecycle_create_has_dependency")

    chapter_id = int(record.get("chapter_id", 0) or 0)
    current = {
        (str(edge.get("source_id")), str(edge.get("target_id")))
        for edge in lifecycle_edges
        if int(nodes.get(edge.get("target_id"), {}).get("metadata", {}).get(
            "chapter_id", 0
        ) or 0) == chapter_id
    }
    declared = {
        (str(row.get("source_id")), str(row.get("target_id")))
        for row in record.get("lifecycle_edges", []) if isinstance(row, dict)
    }
    if current != declared:
        errors.append("record_lifecycle_edges_mismatch")
    validation = record.get("writeback_validation", {})
    if not isinstance(validation, dict) or validation.get("update") != len(current):
        errors.append("record_lifecycle_update_count")


def _validate_inclusion(
    nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]], errors: list[str],
) -> None:
    allowed = {("L1", "L2"), ("L2", "L3"), ("L3", "L4")}
    incoming: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in nodes}
    roots = [node_id for node_id, node in nodes.items() if node.get("level") == "L1"]
    if len(roots) != 1:
        errors.append("inclusion_root_count")
    for edge in edges:
        if edge.get("edge_type") != "INCLUSION":
            continue
        source, target = nodes.get(edge.get("source_id")), nodes.get(edge.get("target_id"))
        if source is None or target is None:
            errors.append("inclusion_missing_node")
            continue
        incoming[target["node_id"]].append(edge)
        if (source.get("level"), target.get("level")) not in allowed:
            errors.append("inclusion_level_mismatch")
    for node_id, node in nodes.items():
        expected = 0 if node.get("level") == "L1" else 1
        if len(incoming[node_id]) != expected:
            errors.append("inclusion_parent_count")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard one M5 collaboration commit")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--chapter", type=int, required=True)
    args = parser.parse_args()
    result = guard_collaboration_commit(args.run_root.resolve(), args.chapter)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
