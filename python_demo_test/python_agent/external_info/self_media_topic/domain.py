"""自媒体选题数据域适配层。

作用：只对数据库热点记录做标题向量聚类，并把联网搜索结果作为补充证据保留。
项目依赖：`bocha_client.py` 提供联网搜索结果；由 `handlers.py` 和包导出层调用。
外部依赖：标题优先使用本地 BGE（`transformers`、`torch`），模型不可用时降级为标准库 TF-IDF；联网搜索通过 Bocha 间接访问。
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .bocha_client import fetch_web_search_sources as _fetch_web_search_sources


EventCluster = Dict[str, Any]
DataSource = Dict[str, Any]
Vector = Any

DEFAULT_CLUSTER_THRESHOLD = 0.86
MAX_CLUSTER_RECORDS = 2000
DEFAULT_EMBEDDING_MODEL_DIR = (
    Path(__file__).resolve().parents[4] / "数据挖掘" / "bge-large-zh-v1.5"
)


def normalize_title(title: str) -> str:
    """清理标题中的 URL、标点和空白，作为向量化输入。"""

    text = re.sub(r"https?://\S+", "", title or "")
    text = re.sub(r"[#【】\[\]（）()《》“”\"'，。！？、:：\s]+", "", text)
    return text.lower()


def split_tags(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in re.split(r"[,，/、\s]+", str(value)) if item.strip()]


def normalize_rank_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rank_records = [item for item in record.get("rank_records", []) or [] if isinstance(item, dict)]
    if record.get("rank_no") is not None:
        rank_records.append(
            {
                "rank_no": record.get("rank_no"),
                "record_time": record.get("record_time")
                or record.get("latest_seen_time")
                or record.get("publish_time"),
            }
        )
    return rank_records


def normalize_source_record(record: Dict[str, Any], source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把一条数据库热点记录归一化为可聚类 item。"""

    title = record.get("title") or record.get("headline") or record.get("representative_title") or record.get("topic")
    if not title:
        return None
    tags: List[str] = []
    for key in ("tags", "tag_names", "news_tags", "tag_name", "category"):
        tags.extend(split_tags(record.get(key)))
    item = {
        "raw_id": record.get("id") or record.get("hot_news_id") or record.get("content_id"),
        "title": str(title),
        "url": record.get("url") or record.get("link"),
        "platform_code": record.get("platform_code") or record.get("platform"),
        "platform_name": record.get("platform_name"),
        "publish_time": record.get("publish_time") or record.get("published_at"),
        "first_seen_time": record.get("first_seen_time"),
        "latest_seen_time": record.get("latest_seen_time") or record.get("record_time"),
        "rank_records": normalize_rank_records(record),
        "tags": list(dict.fromkeys(tags)),
        "source_id": source.get("source_id"),
        "source_type": source.get("source_type"),
        "source_name": source.get("source_name"),
    }
    return {key: value for key, value in item.items() if value not in (None, "", [])}


def _is_database_source(source: Dict[str, Any]) -> bool:
    return str(source.get("source_type") or "").strip().lower() == "database"


def _is_search_source(source: Dict[str, Any]) -> bool:
    return str(source.get("source_type") or "").strip().lower() == "search"


def _title_ngrams(title: str) -> List[str]:
    text = normalize_title(title)
    if not text:
        return []
    if len(text) == 1:
        return [text]
    # 中文标题以字和相邻二元组为特征；英文/数字连续片段也保留为一个特征。
    features = list(text)
    features.extend(text[index : index + 2] for index in range(len(text) - 1))
    features.extend(re.findall(r"[a-z0-9+#.-]+", text))
    return features


def build_tfidf_title_vectors(titles: Sequence[str]) -> List[Vector]:
    """构造轻量 TF-IDF 向量，作为 BGE 不可用时的降级方案。"""

    documents = [_title_ngrams(title) for title in titles]
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))
    total = max(1, len(documents))
    vectors: List[Vector] = []
    for document in documents:
        counts = Counter(document)
        length = max(1, len(document))
        vector = {
            feature: (count / length) * (math.log((total + 1) / (document_frequency[feature] + 1)) + 1.0)
            for feature, count in counts.items()
        }
        vectors.append(vector)
    return vectors


def _embedding_model_dir(model_dir: Optional[str] = None) -> Path:
    configured = str(model_dir or os.getenv("AGENT_EMBEDDING_MODEL_DIR", "")).strip()
    return Path(configured) if configured else DEFAULT_EMBEDDING_MODEL_DIR


@lru_cache(maxsize=2)
def _load_bge_components(model_dir: str) -> Tuple[Any, Any, Any]:
    """懒加载本地 BGE，避免 Agent 启动时立即占用约 1.3GB 模型内存。"""

    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def build_bge_title_vectors(
    titles: Sequence[str],
    model_dir: Optional[str] = None,
    batch_size: int = 16,
) -> List[Vector]:
    """使用本地 BGE 对标题编码，返回已归一化的稠密向量。"""

    import torch
    import torch.nn.functional as functional

    resolved_dir = _embedding_model_dir(model_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"embedding model directory not found: {resolved_dir}")
    tokenizer, model, device = _load_bge_components(str(resolved_dir))
    vectors: List[Vector] = []
    safe_batch_size = max(1, min(int(batch_size or 16), 64))
    with torch.no_grad():
        for start in range(0, len(titles), safe_batch_size):
            batch = [str(title or "") for title in titles[start : start + safe_batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            normalized = functional.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.cpu().tolist())
    return vectors


def build_title_vectors(
    titles: Sequence[str],
    model_dir: Optional[str] = None,
    prefer_bge: bool = True,
) -> Tuple[List[Vector], str]:
    """优先使用本地 BGE；不可用时返回 TF-IDF 向量和降级标识。"""

    if prefer_bge:
        try:
            return build_bge_title_vectors(titles, model_dir=model_dir), "bge-large-zh-v1.5"
        except Exception:
            # 工具边界不能因本地模型缺失而阻断数据库热点分析。
            pass
    return build_tfidf_title_vectors(titles), "tfidf_fallback"


def cosine_similarity(left: Vector, right: Vector) -> float:
    """计算两个稀疏标题向量的余弦相似度。"""

    if not left or not right:
        return 0.0
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
        right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def _merge_ranking_records(items: List[Dict[str, Any]], ranking_records: Iterable[Dict[str, Any]]) -> None:
    """把第二阶段 hot_news_record 结果合并回对应数据库热点。"""

    by_id = {str(item.get("raw_id")): item for item in items if item.get("raw_id") is not None}
    for record in ranking_records:
        item = by_id.get(str(record.get("hot_news_id")))
        if item is None:
            continue
        if record.get("rank_no") is None and record.get("record_time") is None:
            continue
        item.setdefault("rank_records", []).append(
            {"rank_no": record.get("rank_no"), "record_time": record.get("record_time")}
        )


def cluster_database_records(
    records: Sequence[Dict[str, Any]],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    embedding_model_dir: Optional[str] = None,
) -> List[EventCluster]:
    """仅对数据库热点做标题向量两两比较并按阈值合并事件簇。

    当前实现为 O(n^2) 两两比较，使用并查集执行传递合并。
    """

    items = [dict(record) for record in records if isinstance(record, dict) and record.get("title")]
    items = items[:MAX_CLUSTER_RECORDS]
    if not items:
        return []
    threshold_value = max(0.0, min(1.0, float(threshold)))
    vectors, embedding_method = build_title_vectors(
        [str(item.get("title") or "") for item in items],
        model_dir=embedding_model_dir,
    )
    union_find = _UnionFind(len(items))
    edges: List[Tuple[int, int, float]] = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            similarity = cosine_similarity(vectors[left], vectors[right])
            if similarity >= threshold_value:
                edges.append((left, right, similarity))
                union_find.union(left, right)

    edges_by_root: Dict[int, List[float]] = defaultdict(list)
    for left, right, similarity in edges:
        edges_by_root[union_find.find(left)].append(similarity)

    groups: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(items)):
        groups[union_find.find(index)].append(index)

    clusters: List[EventCluster] = []
    for indexes in groups.values():
        cluster_items = [items[index] for index in indexes]
        cluster_items.sort(key=lambda item: str(item.get("latest_seen_time") or ""), reverse=True)
        tags = list(dict.fromkeys(tag for item in cluster_items for tag in item.get("tags", [])))
        source_ids = list(dict.fromkeys(str(item["source_id"]) for item in cluster_items if item.get("source_id")))
        source_types = list(dict.fromkeys(str(item["source_type"]) for item in cluster_items if item.get("source_type")))
        similarities = edges_by_root.get(union_find.find(indexes[0]), [])
        clusters.append(
            {
                "event_id": f"event_{len(clusters) + 1:04d}",
                "cluster_id": len(clusters) + 1,
                "size": len(cluster_items),
                "platform_count": len({item.get("platform_code") for item in cluster_items if item.get("platform_code")}),
                "representative_title": cluster_items[0]["title"],
                "tags": tags,
                "source_ids": source_ids,
                "source_types": source_types,
                "cluster_method": f"title_{embedding_method}_cosine_pairwise",
                "similarity_threshold": threshold_value,
                "max_similarity": max(similarities) if similarities else None,
                "avg_edge_similarity": sum(similarities) / len(similarities) if similarities else None,
                "items": cluster_items,
            }
        )
    clusters.sort(key=lambda item: (item["size"], item["platform_count"], item["max_similarity"] or 0), reverse=True)
    for index, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = index
        cluster["event_id"] = f"event_{index:04d}"
    return clusters


def _collect_database_items(data_sources: Optional[List[DataSource]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    ranking_records: List[Dict[str, Any]] = []
    for source in data_sources or []:
        if not isinstance(source, dict) or not _is_database_source(source):
            continue
        for record in source.get("records") or []:
            if not isinstance(record, dict):
                continue
            if record.get("hot_news_id") is not None and record.get("title") is None:
                ranking_records.append(record)
                continue
            item = normalize_source_record(record, source)
            if item:
                items.append(item)
    _merge_ranking_records(items, ranking_records)
    return items, ranking_records


def _collect_search_supplements(data_sources: Optional[List[DataSource]]) -> List[Dict[str, Any]]:
    supplements: List[Dict[str, Any]] = []
    for source in data_sources or []:
        if not isinstance(source, dict) or not _is_search_source(source):
            continue
        for record in source.get("records") or []:
            if isinstance(record, dict):
                supplements.append(dict(record))
    return supplements


def normalize_data_sources(
    data_sources: Optional[List[DataSource]],
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    embedding_model_dir: Optional[str] = None,
) -> List[EventCluster]:
    """只归一化并聚类数据库数据；搜索数据不会进入事件簇。"""

    database_items, _ = _collect_database_items(data_sources)
    return cluster_database_records(
        database_items,
        threshold=cluster_threshold,
        embedding_model_dir=embedding_model_dir,
    )


def normalize_source_bundle(
    data_sources: Optional[List[DataSource]],
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    embedding_model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """返回数据库事件簇与独立的联网补充证据。"""

    events = normalize_data_sources(
        data_sources,
        cluster_threshold=cluster_threshold,
        embedding_model_dir=embedding_model_dir,
    )
    search_records = _collect_search_supplements(data_sources)
    supplementary_sources = [
        source
        for source in (data_sources or [])
        if isinstance(source, dict) and _is_search_source(source)
    ]
    return {
        "events": events,
        "supplementary_sources": supplementary_sources,
        "search_records": search_records,
        "database_cluster_count": len(events),
        "database_cluster_threshold": max(0.0, min(1.0, float(cluster_threshold))),
    }


def fetch_web_search_sources(
    queries: List[str],
    count: int = 10,
    summary: bool = True,
    freshness: Optional[str] = None,
) -> Dict[str, Any]:
    """联网搜索补充资料，不参与数据库事件聚类。"""

    return _fetch_web_search_sources(
        queries=queries,
        count=count,
        summary=summary,
        freshness=freshness,
    )
