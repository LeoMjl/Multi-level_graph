from __future__ import annotations

import hashlib
from typing import Any

from mlg.graph import EdgeType, NodeLevel, NodeStatus, TaskGraph
from mlg.m5.backend import GenerationResult, TextBackend
from mlg.m5.dataset import ChapterPrompt, han_char_count
from mlg.m5.memory import MemoryContext, MemoryStrategy, parse_json_object
from mlg.m5.rag_memory import EmbeddingBackend
from mlg.m5.taskgraph_checkpoint import dump_taskgraph_memory, load_taskgraph_memory
from mlg.m5.taskgraph_config import PaperDependencyConfig
from mlg.m5.taskgraph_dependency import PaperDependencyBuilder, node_text
from mlg.m5.taskgraph_extract import (
    active_state_registry,
    atomic_facts,
    memory_system_prompt,
    memory_user_prompt,
    select_writeback_registry as bounded_writeback_registry,
    summary_fact,
)
from mlg.m5.taskgraph_prompt_projection import (
    deduplicate_blocks,
    pack_projected_memory,
    project_nodes,
)
from mlg.m5.taskgraph_relation import judge_dependencies


class TaskGraphNovelMemory(MemoryStrategy):
    """M5 mapping: L1 novel, L2 volumes, L3 released chapters, L4 atomic state."""

    name = "taskgraph"
    state_schema = "m5-taskgraph-memory-v6"

    def __init__(
        self,
        embeddings: EmbeddingBackend,
        relationship_backend: TextBackend,
        *,
        token_budget: int = 12000,
        total_volumes: int = 8,
        chapters_per_volume: int = 40,
        volume_briefs: dict[int, str] | None = None,
        dependency_config: PaperDependencyConfig | None = None,
    ) -> None:
        super().__init__(token_budget=token_budget)
        self.embeddings = embeddings
        self.relationship_backend = relationship_backend
        self.total_volumes = total_volumes
        self.chapters_per_volume = chapters_per_volume
        self.volume_briefs = dict(volume_briefs or {})
        self.dependency_builder = PaperDependencyBuilder(dependency_config)
        self.graph = TaskGraph()
        self.root_id = self.graph.add_node(
            NodeLevel.L1, "完成当前逐章释放的长篇小说", 0, "M5",
            hint="Novel", status=NodeStatus.ACTIVE,
        )
        self.volume_by_id: dict[int, str] = {}
        self.chapter_by_id: dict[int, str] = {}
        self.fact_by_key: dict[str, str] = {}
        self.node_vectors: dict[str, list[float]] = {}
        self._create_volume_skeleton()

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        target, audit, api_calls = self.prepare_context_judgment(prompt)
        decisions, judge_result = judge_dependencies(
            self.graph, [(target, audit)], self.relationship_backend,
        )
        api_calls.append(judge_result.to_dict())
        return self.apply_context_decisions(
            prompt, audit, decisions[target], api_calls=api_calls,
        )

    def prepare_context_judgment(
        self, prompt: ChapterPrompt,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        """Create the planned L3 and freeze its three-channel candidate set."""
        if prompt.chapter_id not in self.chapter_by_id:
            self._release_chapter(prompt)
        chapter_node_id = self.chapter_by_id[prompt.chapter_id]
        api_calls: list[dict[str, Any]] = []
        batch = self._encode_missing_nodes(chapter_node_id)
        if batch is not None:
            api_calls.append(batch.call.to_dict())
        chapter_node = self.graph.nodes[chapter_node_id]
        audit = self.dependency_builder.candidates(
            self.graph, chapter_node_id, node_text(chapter_node),
            self.node_vectors[chapter_node_id], self.node_vectors,
        )
        audit["target_phase"] = "planned_l3_before_write"
        return chapter_node_id, audit, api_calls

    def apply_context_decisions(
        self,
        prompt: ChapterPrompt,
        audit: dict[str, Any],
        decisions: list[dict[str, Any]],
        *,
        api_calls: list[dict[str, Any]] | None = None,
    ) -> MemoryContext:
        """Create pre-write edges, then serialize exactly those dependencies."""
        chapter_node_id = self.chapter_by_id[prompt.chapter_id]
        audit = self.dependency_builder.connect_selected(
            self.graph, chapter_node_id, audit, decisions,
            selection_phase="pre_write",
        )

        dependencies = [
            (self.graph.nodes[edge.source_id], edge)
            for edge in self.graph.edges
            if edge.target_id == chapter_node_id
            and edge.edge_type == EdgeType.DEPENDENCY
            and self.graph.nodes[edge.source_id].metadata.get("state_status") != "superseded"
        ]
        dependencies.sort(
            key=lambda item: (
                int(item[1].metadata.get("priority", 10**9)),
                -float(item[1].metadata.get("confidence", 0.0)),
                -int(item[0].metadata.get("importance", 1)),
                -item[0].turn_index,
                item[0].node_id,
            ),
        )
        local_chapter = prompt.chapter_id - 1 if prompt.chapter_id > 1 else None
        method_dependencies = [
            item for item in dependencies
            if local_chapter is None
            or int(item[0].metadata.get("chapter_id", 0) or 0) != local_chapter
        ]
        excluded_local_nodes = [
            node.node_id for node, _ in dependencies
            if local_chapter is not None
            and int(node.metadata.get("chapter_id", 0) or 0) == local_chapter
        ]
        projected = project_nodes(
            [node for node, _ in method_dependencies], self.volume_briefs,
        )
        deduplicated = deduplicate_blocks(projected)
        packed = pack_projected_memory(
            deduplicated, self.token_budget,
        )
        selected_nodes = list(dict.fromkeys(packed.selected_node_ids))
        selected_set = set(selected_nodes)
        selected_chapters: list[int] = []
        for node, edge in method_dependencies:
            if node.node_id not in selected_set:
                continue
            chapter_id = int(node.metadata.get("chapter_id", 0))
            edge.metadata["used_in_prompt"] = True
            edge.metadata["prompt_use_turn"] = prompt.chapter_id
            node.metadata["prompt_use_count"] = int(
                node.metadata.get("prompt_use_count", 0)
            ) + 1
            node.metadata["last_prompt_use_turn"] = prompt.chapter_id
            source_chapter = int(node.metadata.get("chapter_id", 0) or 0)
            if (
                source_chapter
                and source_chapter < prompt.chapter_id
                and not edge.metadata.get("ref_counted")
            ):
                node.metadata["ref_count"] = int(
                    node.metadata.get("ref_count", 0)
                ) + 1
                edge.metadata["ref_counted"] = True
            if chapter_id:
                selected_chapters.append(chapter_id)
        for row in audit.get("created", []):
            row["used_in_prompt"] = str(row.get("source_id", "")) in selected_set
        return MemoryContext(packed.text, sorted(set(selected_chapters)), {
            "released_l2": self.volume_by_id[prompt.volume_id],
            "released_l3": chapter_node_id,
            "selected_node_ids": selected_nodes,
            "memory_tokens": packed.memory_tokens,
            "memory_blocks": len(packed.blocks),
            "deduplicated_count": len(projected) - len(deduplicated),
            "dropped_count": packed.dropped_count,
            "excluded_local_node_ids": excluded_local_nodes,
            "graph_nodes": len(self.graph.nodes),
            "graph_edges": len(self.graph.edges),
            "api_calls": api_calls or [],
        })

    def observe(self, prompt: ChapterPrompt, text: str, backend: TextBackend) -> list[GenerationResult]:
        registry = list(self.writeback_registry(text).rows)
        result = backend.generate(
            memory_system_prompt(registry_provided=True),
            memory_user_prompt(prompt, text, registry),
            purpose="memory",
        )
        payload = parse_json_object(result.text)
        facts, targets, build_call = self.prepare_observation_judgment(
            prompt, payload, source_text=text,
        )
        self.apply_observation_decisions(prompt, facts, targets, {})
        return [result, build_call]

    def prepare_observation_judgment(
        self,
        prompt: ChapterPrompt,
        payload: dict[str, Any],
        *,
        source_text: str | None = None,
        parsed_facts: list[dict[str, Any]] | None = None,
    ) -> tuple[
        list[dict[str, Any]], list[tuple[str, dict[str, Any]]], GenerationResult,
    ]:
        """Create validated L4 versions; post-write model candidates do not exist."""
        chapter_node_id = self.chapter_by_id[prompt.chapter_id]
        if source_text is None or not source_text.strip():
            raise ValueError("Completed L3 representation requires committed chapter text")
        if not str(payload.get("summary", "")).strip():
            raise ValueError("Completed L3 representation requires a non-empty result summary")
        registry_keys = {row["key"] for row in self.active_state_registry()}
        extracted = (
            [dict(fact) for fact in parsed_facts]
            if parsed_facts is not None
            else atomic_facts(
                prompt, payload, source_text=source_text,
                registered_keys=registry_keys,
            )
        )
        facts = [summary_fact(prompt, payload), *extracted]
        facts[0]["_chapter_text_sha256"] = hashlib.sha256(
            source_text.encode("utf-8")
        ).hexdigest()
        facts[0]["_chapter_han_chars"] = han_char_count(source_text)
        l3_result_text = "\n".join(("L3", str(facts[0]["value"]).strip()))
        plans: list[tuple[dict[str, Any], str | None]] = []
        text_limit = _embedding_text_limit(self.embeddings)
        if len(l3_result_text) > text_limit:
            raise ValueError(
                f"L3 result exceeds embedding text limit ({text_limit} chars)"
            )
        for fact in facts:
            prior = self._validate_lifecycle_fact(fact)
            if len(_planned_l4_text(fact)) > text_limit:
                raise ValueError(
                    f"L4 fact exceeds embedding text limit ({text_limit} chars): "
                    f"{fact['key']}"
                )
            plans.append((fact, prior))
        embedding_texts = [
            l3_result_text,
            *(_planned_l4_text(fact) for fact, _ in plans),
        ]
        batch = self.embeddings.embed(embedding_texts, purpose="memory_build")
        if len(batch.vectors) != len(facts) + 1:
            raise RuntimeError("TaskGraph state embedding returned the wrong vector count")
        _validate_vector_dimensions(batch.vectors, self.node_vectors)

        node_ids: list[str] = []
        for fact, prior in plans:
            fact["prior_node_id"] = prior
            fact["state_version"] = (
                int(self.graph.nodes[prior].metadata.get("state_version", 1)) + 1
                if prior else 1
            )
            node_id = self._add_l4(chapter_node_id, prompt, fact)
            fact["node_id"] = node_id
            node_ids.append(node_id)
        self.node_vectors[chapter_node_id] = batch.vectors[0]
        self.node_vectors.update(zip(node_ids, batch.vectors[1:]))
        return facts, [], batch.call

    def apply_observation_decisions(
        self,
        prompt: ChapterPrompt,
        facts: list[dict[str, Any]],
        targets: list[tuple[str, dict[str, Any]]],
        decisions: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Commit deterministic stable-key version edges and lifecycle transitions."""
        chapter_node_id = self.chapter_by_id[prompt.chapter_id]
        if targets or decisions:
            raise RuntimeError("Post-write model dependency judgments are disabled")
        created: list[dict[str, Any]] = []
        for fact in facts:
            node_id = str(fact["node_id"])
            prior = fact.get("prior_node_id")
            if fact["operation"] == "update":
                if not prior or self.fact_by_key.get(fact["key"]) != prior:
                    raise RuntimeError(f"Stale lifecycle update for {fact['key']}")
                metadata = {
                    "method": "stable_key_lifecycle_v1",
                    "selection_phase": "state_lifecycle",
                    "dependency_type": "state_continuity",
                    "confidence": 1.0,
                    "priority": 1,
                    "reason": "同一stable key的确定性版本连续边",
                    "stable_key_match": True,
                    "lifecycle_operation": "update",
                    "state_key": fact["key"],
                    "used_in_prompt": False,
                    "ref_counted": False,
                }
                self.graph.add_edge(prior, node_id, EdgeType.DEPENDENCY, metadata)
                self.graph.nodes[prior].metadata.update({
                    "state_status": "superseded",
                    "superseded_by": node_id,
                    "superseded_at_chapter": prompt.chapter_id,
                })
                created.append({
                    "source_id": prior, "target_id": node_id,
                    "state_key": fact["key"], "dependency_type": "state_continuity",
                })
            self.fact_by_key[fact["key"]] = node_id

        self.graph.nodes[chapter_node_id].status = NodeStatus.DONE
        chapter_node = self.graph.nodes[chapter_node_id]
        result_summary = str(facts[0]["value"]).strip()
        chapter_node.value = result_summary
        chapter_node.metadata.update({
            "lifecycle_phase": "observed_after_write",
            "result_summary": result_summary,
            "result_text_sha256": facts[0].pop("_chapter_text_sha256"),
            "result_han_chars": facts[0].pop("_chapter_han_chars"),
            "result_source": "committed_chapter_with_extractor_summary",
            "embedding_representation": "actual_completed_l3",
        })
        volume_id = prompt.volume_id
        volume_node = self.graph.nodes[self.volume_by_id[volume_id]]
        volume_node.metadata["released_chapters"] = int(volume_node.metadata.get("released_chapters", 0)) + 1
        if prompt.chapter_id % self.chapters_per_volume == 0:
            volume_node.status = NodeStatus.DONE
            volume_node.metadata["expanded"] = True
        if prompt.chapter_id >= self.total_volumes * self.chapters_per_volume:
            self.graph.nodes[self.root_id].status = NodeStatus.DONE
        return created

    def _validate_lifecycle_fact(self, fact: dict[str, Any]) -> str | None:
        key = str(fact.get("key", ""))
        operation = str(fact.get("operation", ""))
        prior = self.fact_by_key.get(key)
        if operation == "create":
            if prior is not None or fact.get("prior_key"):
                raise ValueError(f"Invalid create lifecycle for {key}")
            return None
        if operation != "update":
            raise ValueError(f"Invalid lifecycle operation for {key}: {operation}")
        if prior is None or fact.get("prior_key") != key:
            raise ValueError(f"Invalid update lifecycle for {key}")
        return prior

    def _create_volume_skeleton(self) -> None:
        for volume_id in range(1, self.total_volumes + 1):
            brief = self.volume_briefs.get(volume_id, f"完成第{volume_id}卷公开目标")
            node_id = self.graph.add_node(
                NodeLevel.L2, f"推进第{volume_id}卷长期目标", 0, f"M5/V{volume_id}",
                hint="Volume", status=NodeStatus.PENDING, value=brief,
                metadata={"volume_id": volume_id, "expanded": False, "released_chapters": 0},
            )
            self.graph.add_edge(self.root_id, node_id, EdgeType.INCLUSION)
            self.volume_by_id[volume_id] = node_id

    def _release_chapter(self, prompt: ChapterPrompt) -> None:
        volume_id = self.volume_by_id[prompt.volume_id]
        self.graph.nodes[volume_id].status = NodeStatus.ACTIVE
        chapter_id = self.graph.add_node(
            NodeLevel.L3, f"撰写第{prompt.chapter_id}章《{prompt.title}》：{prompt.chapter_goal}",
            prompt.chapter_id, f"M5/V{prompt.volume_id}/C{prompt.chapter_id}",
            hint="Chapter", status=NodeStatus.PENDING, value="\n".join([
                *(f"必须包含：{item}" for item in prompt.must_include),
                *(f"必须避免：{item}" for item in prompt.must_avoid),
            ]),
            metadata={
                "chapter_id": prompt.chapter_id,
                "volume_id": prompt.volume_id,
                "lifecycle_phase": "planned_before_write",
                "ref_count": 0,
                "prompt_use_count": 0,
            },
        )
        self.graph.add_edge(volume_id, chapter_id, EdgeType.INCLUSION)
        self.graph.nodes[chapter_id].status = NodeStatus.ACTIVE
        self.chapter_by_id[prompt.chapter_id] = chapter_id

    def _add_l4(self, chapter_id: str, prompt: ChapterPrompt, fact: dict[str, Any]) -> str:
        node_id = self.graph.add_node(
            NodeLevel.L4, str(fact["key"]), prompt.chapter_id,
            f"M5/V{prompt.volume_id}/C{prompt.chapter_id}/{fact['key']}",
            hint="State", status=NodeStatus.DONE, value=str(fact["value"]),
            metadata={
                "chapter_id": prompt.chapter_id, "volume_id": prompt.volume_id,
                "state_key": fact["key"], "entities": fact["entities"],
                "importance": fact["importance"],
                "state_status": fact.get("final_status", fact.get("status", "active")),
                "state_version": int(fact.get("state_version", 1)),
                "lifecycle_operation": fact.get("operation", "create"),
                "prior_key": fact.get("prior_key"),
                "prior_node_id": fact.get("prior_node_id"),
                "source_quote": fact.get("source_quote", ""),
                "source_quote_valid": fact.get("source_quote_valid"), "ref_count": 0,
                "prompt_use_count": 0,
            },
        )
        self.graph.add_edge(chapter_id, node_id, EdgeType.INCLUSION)
        return node_id

    def _encode_missing_nodes(self, target_id: str):
        target = self.graph.nodes[target_id]
        node_ids = [
            node_id for node_id, node in self.graph.nodes.items()
            if node_id not in self.node_vectors
            and _eligible_for_embedding(node, target)
        ]
        if not node_ids:
            return None
        batch = self.embeddings.embed(
            [node_text(self.graph.nodes[node_id]) for node_id in node_ids],
            purpose="memory_build",
        )
        if len(batch.vectors) != len(node_ids):
            raise RuntimeError("TaskGraph node encoding returned the wrong vector count")
        _validate_vector_dimensions(batch.vectors, self.node_vectors)
        self.node_vectors.update(zip(node_ids, batch.vectors))
        return batch

    def active_state_registry(self) -> list[dict[str, Any]]:
        return active_state_registry(self.graph, self.fact_by_key)

    def writeback_registry(self, source_text: str):
        return bounded_writeback_registry(
            self.graph, self.fact_by_key, source_text,
        )

    def to_state(self) -> dict[str, Any]:
        return dump_taskgraph_memory(self, super().to_state())

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        load_taskgraph_memory(self, state)


def _eligible_for_embedding(node, target) -> bool:
    """Limit external encoding to released L2-L4 history and the current target."""
    if node.level == NodeLevel.L1:
        return False
    if node.status in {NodeStatus.PENDING, NodeStatus.DROPPED}:
        return False
    if node.metadata.get("state_status") == "superseded":
        return False
    if node.turn_index > target.turn_index:
        return False
    target_volume = int(target.metadata.get("volume_id", 0) or 0)
    node_volume = int(node.metadata.get("volume_id", 0) or 0)
    if node.level == NodeLevel.L2 and target_volume and node_volume > target_volume:
        return False
    target_chapter = int(target.metadata.get("chapter_id", 0) or 0)
    node_chapter = int(node.metadata.get("chapter_id", 0) or 0)
    return not (node_chapter and target_chapter and node_chapter > target_chapter)


def _embedding_text_limit(embeddings: EmbeddingBackend) -> int:
    configured = getattr(embeddings, "max_input_chars", 1600)
    try:
        return max(1, min(1600, int(configured)))
    except (TypeError, ValueError):
        return 1600


def _planned_l4_text(fact: dict[str, Any]) -> str:
    entities = "、".join(str(item) for item in fact.get("entities", []))
    return "\n".join(
        item for item in ("L4", str(fact["key"]), str(fact["value"]), entities)
        if item
    )


def _validate_vector_dimensions(
    vectors: list[list[float]], existing: dict[str, list[float]],
) -> None:
    existing_dimensions = {len(vector) for vector in existing.values()}
    if len(existing_dimensions) > 1:
        raise RuntimeError("Existing TaskGraph embedding index has mixed dimensions")
    expected = next(iter(existing_dimensions), len(vectors[0]) if vectors else 0)
    if expected <= 0 or any(len(vector) != expected for vector in vectors):
        raise RuntimeError(
            f"TaskGraph embedding dimension changed across batches; expected {expected}"
        )
