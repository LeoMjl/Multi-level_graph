from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlg.graph.task_graph import EdgeType, NodeLevel, NodeStatus, TaskGraph


TOP_K = 10


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized(vectors: Any) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


@dataclass
class RetrievalIndex:
    dataset: str
    documents: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    chunk_vectors: np.ndarray

    def __post_init__(self) -> None:
        self.chunk_vectors = normalized(self.chunk_vectors)
        self.chunk_by_id = {row["chunk_id"]: index for index, row in enumerate(self.chunks)}
        self.doc_by_id = {row["doc_id"]: row for row in self.documents}
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            grouped[chunk["doc_id"]].append(index)
        self.doc_chunks = {key: sorted(value, key=lambda i: self.chunks[i]["chunk_index"]) for key, value in grouped.items()}
        self.doc_ids = list(self.doc_by_id)
        self.doc_vectors = normalized(np.vstack([
            np.mean(self.chunk_vectors[self.doc_chunks[doc_id]], axis=0) for doc_id in self.doc_ids
        ]))
        self.doc_position = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        self.dependencies = self._semantic_dependencies()

    def _semantic_dependencies(self, count: int = 4) -> dict[str, list[tuple[str, float]]]:
        similarities = self.doc_vectors @ self.doc_vectors.T
        output: dict[str, list[tuple[str, float]]] = {}
        for index, doc_id in enumerate(self.doc_ids):
            order = np.argsort(-similarities[index])
            output[doc_id] = [
                (self.doc_ids[item], float(np.clip(similarities[index, item], -1.0, 1.0)))
                for item in order if item != index
            ][:count]
        return output

    def build_graph(self) -> TaskGraph:
        graph = TaskGraph()
        root = graph.add_node(
            NodeLevel.L1, f"{self.dataset} document QA", 0, self.dataset,
            hint="Corpus", status=NodeStatus.ACTIVE, metadata={"dataset": self.dataset},
        )
        groups: dict[str, str] = {}
        doc_nodes: dict[str, str] = {}
        for document in self.documents:
            group = str(document.get("category") or "uncategorized")
            if group not in groups:
                groups[group] = graph.add_node(NodeLevel.L2, group, 0, f"{self.dataset}/{group}", hint="Group")
                graph.add_edge(root, groups[group], EdgeType.INCLUSION)
            doc_id = document["doc_id"]
            doc_nodes[doc_id] = graph.add_node(
                NodeLevel.L3, str(document.get("title") or doc_id), 0,
                f"{self.dataset}/{group}/{doc_id}", hint="Doc", metadata={"doc_id": doc_id},
            )
            graph.add_edge(groups[group], doc_nodes[doc_id], EdgeType.INCLUSION)
            previous = ""
            for chunk_index in self.doc_chunks[doc_id]:
                chunk = self.chunks[chunk_index]
                node = graph.add_node(
                    NodeLevel.L4, chunk["fact_text"], int(chunk["chunk_index"]),
                    f"{self.dataset}/{group}/{doc_id}/{chunk['chunk_index']}", hint="Chunk",
                    metadata={"doc_id": doc_id, "chunk_id": chunk["chunk_id"]},
                )
                graph.add_edge(doc_nodes[doc_id], node, EdgeType.INCLUSION)
                if previous:
                    graph.add_edge(previous, node, EdgeType.MAINLINE)
                previous = node
        for source, neighbors in self.dependencies.items():
            for target, score in neighbors:
                graph.add_edge(doc_nodes[source], doc_nodes[target], EdgeType.DEPENDENCY, {"cosine": score})
        return graph

    def flat(
        self, query_vector: Any, *, doc_id: str = "", k: int = TOP_K, policy: str = "dense",
    ) -> list[dict[str, Any]]:
        qvec = normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        candidates = self.doc_chunks[doc_id] if doc_id else list(range(len(self.chunks)))
        scores = self.chunk_vectors[candidates] @ qvec
        order = np.argsort(-scores)[:k]
        return [self._result(candidates[pos], float(scores[pos]), policy) for pos in order]

    def taskgraph(self, query_vector: Any, *, doc_id: str = "", k: int = TOP_K) -> list[dict[str, Any]]:
        qvec = normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        candidates = self.doc_chunks[doc_id] if doc_id else list(range(len(self.chunks)))
        dense = self.chunk_vectors[candidates] @ qvec
        doc_scores = self.doc_vectors @ qvec
        top_seed_positions = np.argsort(-dense)[: min(16, len(candidates))]
        bonuses: dict[int, float] = defaultdict(float)
        for rank, position in enumerate(top_seed_positions):
            index = candidates[position]
            seed = self.chunks[index]
            bonuses[index] = max(bonuses[index], 1.0 - rank / 32.0)
            ordered = self.doc_chunks[seed["doc_id"]]
            local = ordered.index(index)
            for distance in (1, 2):
                for neighbor_pos in (local - distance, local + distance):
                    if 0 <= neighbor_pos < len(ordered):
                        bonuses[ordered[neighbor_pos]] = max(bonuses[ordered[neighbor_pos]], 0.6 / distance)
            if not doc_id:
                for neighbor_doc, edge_score in self.dependencies[seed["doc_id"]]:
                    neighbor_indices = self.doc_chunks[neighbor_doc]
                    neighbor_dense = self.chunk_vectors[neighbor_indices] @ qvec
                    best = neighbor_indices[int(np.argmax(neighbor_dense))]
                    bonuses[best] = max(bonuses[best], max(0.0, edge_score) * 0.5)
        pool = sorted(set(candidates if doc_id else bonuses) | set(candidates[i] for i in top_seed_positions))
        ranked = []
        for index in pool:
            chunk_score = float(self.chunk_vectors[index] @ qvec)
            document_score = float(doc_scores[self.doc_position[self.chunks[index]["doc_id"]]])
            score = 0.65 * chunk_score + 0.20 * document_score + 0.15 * bonuses[index]
            ranked.append((score, index))
        ranked.sort(reverse=True)
        selected, per_doc = [], defaultdict(int)
        cap = k if doc_id else 3
        for score, index in ranked:
            current_doc = self.chunks[index]["doc_id"]
            if per_doc[current_doc] >= cap:
                continue
            selected.append(self._result(index, score, "taskgraph_hybrid"))
            per_doc[current_doc] += 1
            if len(selected) == k:
                break
        return selected

    def _result(self, index: int, score: float, policy: str) -> dict[str, Any]:
        chunk = self.chunks[index]
        return {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "score": score, "policy": policy, "text": chunk["text"]}
