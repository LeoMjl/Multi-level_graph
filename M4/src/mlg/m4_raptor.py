from __future__ import annotations

import copy
import hashlib
import json
import logging
import pickle
import random
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

from mlg.config import PROJECT_ROOT
from mlg.m4_model import M4Model
from mlg.m4_retrieval import normalized


RAPTOR_ROOT = PROJECT_ROOT / "third_party" / "official" / "raptor"
RAPTOR_COMMIT = "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767"
CONTEXT_TOKENS = 8192
TOP_K = 10
RAPTOR_PROTOCOL = "m4-raptor-v4-bounded-recovery"


def _normalize_summary(value: Any) -> str:
    """Unwrap providers that encode a JSON answer inside another JSON field."""
    current = value
    for _ in range(4):
        if isinstance(current, dict):
            current = current.get("summary") or current.get("answer") or ""
            continue
        text = str(current or "").strip()
        if not text.startswith("{"):
            return text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(parsed, dict):
            return text
        current = parsed
    return str(current or "").strip() if not isinstance(current, dict) else ""


def _is_truncated_summary(value: Any) -> bool:
    if isinstance(value, dict):
        value = value.get("summary") or value.get("answer") or ""
    text = str(value or "").strip()
    if not text.startswith(("{\"summary\":", "{\"answer\":")):
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return True
    return False


def _recover_truncated_summary(value: Any, max_tokens: int) -> str:
    if isinstance(value, dict):
        value = value.get("summary") or value.get("answer") or ""
    text = str(value or "").strip()
    for prefix in ('{"summary":"', '{"answer":"'):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.endswith("\\"):
        text = text[:-1]
    try:
        text = json.loads(f'"{text}"')
    except json.JSONDecodeError:
        text = text.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")
    tokenizer = tiktoken.get_encoding("o200k_base")
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        text = tokenizer.decode(tokens[:max_tokens])
    sentence_end = max(text.rfind("."), text.rfind("?"), text.rfind("!"))
    if sentence_end >= max(20, len(text) // 3):
        text = text[: sentence_end + 1]
    return text.strip()


def _load_official():
    if str(RAPTOR_ROOT) not in sys.path:
        sys.path.insert(0, str(RAPTOR_ROOT))
    from raptor import BaseEmbeddingModel, BaseSummarizationModel
    import raptor.cluster_utils as cluster_utils
    from raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
    from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig
    from raptor.tree_structures import Node, Tree
    if not getattr(cluster_utils, "_m4_stable_gmm", False):
        cluster_utils.GaussianMixture = partial(cluster_utils.GaussianMixture, reg_covar=1e-4)
        cluster_utils._m4_stable_gmm = True
    if not getattr(cluster_utils, "_m4_stable_small_clusters", False):
        original_clustering = cluster_utils.RAPTOR_Clustering.perform_clustering

        def stable_clustering(
            nodes, embedding_model_name, max_length_in_cluster=3500,
            tokenizer=tiktoken.get_encoding("cl100k_base"), **kwargs,
        ):
            if len(nodes) <= 4:
                groups, current, used = [], [], 0
                for node in nodes:
                    length = len(tokenizer.encode(node.text))
                    if current and used + length > max_length_in_cluster:
                        groups.append(current)
                        current, used = [], 0
                    current.append(node)
                    used += length
                if current:
                    groups.append(current)
                return groups
            return original_clustering(
                nodes, embedding_model_name, max_length_in_cluster,
                tokenizer=tokenizer, **kwargs,
            )

        cluster_utils.RAPTOR_Clustering.perform_clustering = staticmethod(stable_clustering)
        cluster_utils._m4_stable_small_clusters = True
    logging.getLogger().setLevel(logging.WARNING)
    return BaseEmbeddingModel, BaseSummarizationModel, ClusterTreeBuilder, ClusterTreeConfig, TreeRetriever, TreeRetrieverConfig, Node, Tree


class _JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, Any] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.rows[row["key"]] = row["value"]

    def get(self, text: str):
        return self.rows.get(hashlib.sha256(text.encode("utf-8")).hexdigest())

    def put(self, text: str, value: Any, *, overwrite: bool = False) -> None:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self.rows and not overwrite:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        self.rows[key] = value


def _adapters(model: M4Model, cache_dir: Path, known: dict[str, list[float]] | None = None):
    BaseEmbeddingModel, BaseSummarizationModel, *_ = _load_official()
    embedding_cache = _JsonlCache(cache_dir / "embeddings.jsonl")
    summary_cache = _JsonlCache(cache_dir / "summaries.jsonl")
    recovery_cache = _JsonlCache(cache_dir / "summary_recoveries.jsonl")
    known = known or {}

    class Embedding(BaseEmbeddingModel):
        def create_embedding(self, text):
            if text in known:
                return known[text]
            cached = embedding_cache.get(text)
            if cached is not None:
                return cached
            vectors = model.embed_texts([text], phase="memory_build")
            if len(vectors) != 1:
                raise RuntimeError(model._last_embedding_error or "RAPTOR embedding failed")
            embedding_cache.put(text, vectors[0])
            return vectors[0]

    class Summarizer(BaseSummarizationModel):
        def summarize(self, context, max_tokens=150):
            cached = summary_cache.get(context)
            if cached is not None and not _is_truncated_summary(cached):
                summary = _normalize_summary(cached)
                if not summary:
                    raise RuntimeError("RAPTOR cached summarizer returned no summary")
                return summary
            result = model.chat_json(
                "Summarize the supplied passages while preserving names, events, dates, comparisons, and causal links. "
                'Use one concise paragraph of at most 50 words. Return JSON as {"summary":"..."}.',
                context,
                max_tokens=max(800, max_tokens * 8),
                phase="memory_build",
            )
            if _is_truncated_summary(result):
                summary = _recover_truncated_summary(result, max_tokens)
                recovery_cache.put(context, {"strategy": "bounded_complete_prefix", "summary": summary})
            else:
                summary = _normalize_summary(result)
            if not summary:
                candidates = [value for key, value in result.items() if not key.startswith("_") and isinstance(value, str)]
                summary = candidates[0].strip() if len(candidates) == 1 else ""
            if result.get("_error") or not summary:
                raise RuntimeError(str(result.get("_error") or "RAPTOR summarizer returned no summary"))
            summary_cache.put(context, summary, overwrite=cached is not None)
            return summary

    return Embedding(), Summarizer()


def build_raptor(
    dataset: str, documents: list[dict], chunks: list[dict], vectors: Any,
    model: M4Model, artifact_dir: Path,
) -> dict[str, Any]:
    _, _, ClusterTreeBuilder, ClusterTreeConfig, _, _, Node, Tree = _load_official()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[dict, list[float]]]] = defaultdict(list)
    key_for_doc = {row["doc_id"]: row["doc_id"] if dataset == "quality" else "unified" for row in documents}
    for chunk, vector in zip(chunks, vectors):
        grouped[key_for_doc[chunk["doc_id"]]].append((chunk, np.asarray(vector).tolist()))
    built = skipped = nodes = 0
    for number, key in enumerate(sorted(grouped), 1):
        tree_path = artifact_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.pkl"
        meta_path = tree_path.with_suffix(".json")
        if tree_path.exists() and meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("protocol") == RAPTOR_PROTOCOL:
                skipped += 1
                continue
        rows = sorted(grouped[key], key=lambda item: item[0]["chunk_id"])
        known = {row["text"]: vector for row, vector in rows}
        embedding, summarizer = _adapters(model, artifact_dir / "model_cache", known)
        config = ClusterTreeConfig(
            tokenizer=tiktoken.get_encoding("o200k_base"), max_tokens=512 if dataset == "quality" else 1024,
            num_layers=5, summarization_length=100, summarization_model=summarizer,
            embedding_models={"M4": embedding}, cluster_embedding_model="M4",
        )
        builder = ClusterTreeBuilder(config)
        leaves = {i: Node(row["text"], i, set(), {"M4": vector}) for i, (row, vector) in enumerate(rows)}
        all_nodes = copy.deepcopy(leaves)
        layers = {0: list(all_nodes.values())}
        random.seed(224)
        np.random.seed(224)
        roots = builder.construct_tree(all_nodes, all_nodes, layers, use_multithreading=False)
        tree = Tree(all_nodes, roots, leaves, builder.num_layers, layers)
        with tree_path.open("wb") as handle:
            pickle.dump(tree, handle)
        meta_path.write_text(json.dumps({
            "key": key, "dataset": dataset, "leaf_chunk_ids": [row["chunk_id"] for row, _ in rows],
            "nodes": len(tree.all_nodes), "layers": tree.num_layers + 1,
            "protocol": RAPTOR_PROTOCOL,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        built += 1
        nodes += len(tree.all_nodes)
        if number % 20 == 0 or number == len(grouped):
            print(f"[M4] RAPTOR {dataset}: trees {number}/{len(grouped)}", flush=True)
    return {"trees": len(grouped), "built": built, "reused": skipped, "new_nodes": nodes, "path": str(artifact_dir)}


def retrieve_raptor(
    dataset: str, queries: list[dict], query_vectors: Any, artifact_dir: Path,
) -> list[list[dict[str, Any]]]:
    BaseEmbeddingModel, _, _, _, TreeRetriever, TreeRetrieverConfig, _, _ = _load_official()
    vector_by_question = {
        row["question"]: np.asarray(vector).tolist() for row, vector in zip(queries, query_vectors)
    }

    class QueryEmbedding(BaseEmbeddingModel):
        def create_embedding(self, text):
            try:
                return vector_by_question[text]
            except KeyError as exc:
                raise KeyError("RAPTOR received an unprepared query") from exc

    config = TreeRetrieverConfig(
        tokenizer=tiktoken.get_encoding("o200k_base"), top_k=TOP_K,
        selection_mode="top_k", context_embedding_model="M4", embedding_model=QueryEmbedding(),
    )
    loaded: dict[str, tuple[Any, dict[int, int]]] = {}
    output = []
    for query in queries:
        key = query["doc_id"] if dataset == "quality" else "unified"
        if key not in loaded:
            stem = hashlib.sha256(key.encode()).hexdigest()[:20]
            with (artifact_dir / f"{stem}.pkl").open("rb") as handle:
                tree = pickle.load(handle)
            node_layer = {
                node.index: layer for layer, layer_nodes in tree.layer_to_nodes.items() for node in layer_nodes
            }
            loaded[key] = (TreeRetriever(config, tree), node_layer)
        retriever, node_layer = loaded[key]
        selected, _ = retriever.retrieve_information_collapse_tree(
            query["question"], top_k=TOP_K, max_tokens=CONTEXT_TOKENS,
        )
        qvec = normalized(np.asarray(vector_by_question[query["question"]]).reshape(1, -1))[0]
        rows = []
        for node in selected:
            score = float(normalized(np.asarray(node.embeddings["M4"]).reshape(1, -1))[0] @ qvec)
            rows.append({
                "chunk_id": f"raptor:{key}:node:{node.index}", "doc_id": query.get("doc_id", ""),
                "score": score, "policy": "official_raptor_collapsed", "layer": node_layer[node.index],
                "text": node.text,
            })
        output.append(rows)
    return output
