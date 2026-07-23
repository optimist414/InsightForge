"""自媒体选题工具执行适配层。

作用：把 Agent 输入解析为业务函数调用，并将结果或异常包装成统一 ToolResult。
项目依赖：`schemas.py` 的协议由 `registry.py` 注册；业务逻辑来自画像、数据源和 proposal 模块。
外部依赖：画像阶段可调用 DeepSeek，数据阶段可调用 Java SQL 和 Bocha，但本文件不直接管理连接。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict

from ...view.tooling import ToolContext, ToolResult
from . import domain
from .bilibili_creator import execute_bilibili_creator_videos
from .creator_profile import build_creator_profile as build_creator_profile_fallback
from .creator_profile_prompt import generate_creator_profile_with_llm, parse_creator_profile_response
from .data_sources import fetch_hot_news
from .proposal_metrics import batch_semantic_fit_score, check_risk_evidence, plan_content_angles, score_event


Handler = Callable[[Dict[str, Any], ToolContext], Dict[str, Any]]


def to_tool_result(payload: Dict[str, Any]) -> ToolResult:
    """把业务结果统一序列化成模型可读文本，并在 metadata 保留结构化结果。"""

    return ToolResult(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        metadata={"result": payload},
    )


def to_error_result(exc: Exception) -> ToolResult:
    """工具边界统一兜底，避免异常穿透到 Engine loop。"""

    return ToolResult(
        content=f"self-media topic tool failed: {exc}",
        is_error=True,
        metadata={"error_type": exc.__class__.__name__},
    )


def wrap_handler(handler: Handler):
    """把业务 handler 适配为 BuiltinTool 需要的 ToolResult handler。"""

    def execute(input: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            return to_tool_result(handler(input, ctx))
        except Exception as exc:  # noqa: BLE001 - tool boundary must convert any failure to ToolResult.
            return to_error_result(exc)

    return execute


def normalize_source_bundle(input: Dict[str, Any], _ctx: ToolContext) -> Dict[str, Any]:
    """source_bundle_normalizer 的执行函数。"""

    return domain.normalize_source_bundle(
        input.get("data_sources") or [],
        cluster_threshold=float(input.get("cluster_threshold") or domain.DEFAULT_CLUSTER_THRESHOLD),
    )


def fetch_web_search(input: Dict[str, Any], _ctx: ToolContext) -> Dict[str, Any]:
    """web_search_fetcher 的执行函数。"""

    queries = [str(query).strip() for query in input.get("queries") or [] if str(query).strip()]
    return domain.fetch_web_search_sources(
        queries=queries,
        count=int(input.get("count") or 10),
        summary=bool(input.get("summary", True)),
        freshness=input.get("freshness"),
    )


def build_creator_profile(input: Dict[str, Any], _ctx: ToolContext) -> Dict[str, Any]:
    """采集创作者证据、格式化 history，再调用画像 LLM。"""

    history = [row for row in (input.get("history") or [])[:50] if isinstance(row, dict)]
    explicit = input.get("explicit_profile") if isinstance(input.get("explicit_profile"), dict) else {}
    creator_url = str(input.get("creator_url") or "").strip()
    platform = str(input.get("platform") or ("bilibili" if creator_url else ""))
    creator_id = str(input.get("creator_id") or "")
    llm_output = str(input.get("llm_output") or "").strip()
    collection_info: Dict[str, Any] = {}

    if not history and not llm_output and platform.lower() == "bilibili" and (creator_id or creator_url):
        collection_result = execute_bilibili_creator_videos(
            {
                "uid": creator_id,
                "creator_url": creator_url,
                "limit": min(int(input.get("limit") or 50), 50),
                "include_details": input.get("include_details") is True,
                "timeout_ms": input.get("timeout_ms"),
                "max_attempts": input.get("max_attempts"),
            },
            _ctx,
        )
        if collection_result.is_error:
            raise ValueError(f"创作者数据采集失败：{collection_result.content}")
        payload = (collection_result.metadata or {}).get("result") or {}
        history = normalize_creator_history(payload.get("videos") or [])
        collection_info = {
            "source": "bilibili_creator_videos",
            "platform": "bilibili",
            "sample_count": len(history),
            "requested_limit": min(int(input.get("limit") or 50), 50),
        }
        if not history:
            raise ValueError("创作者数据采集成功，但没有可用于画像的作品样本")

    if not history and not explicit and not llm_output:
        raise ValueError(
            "creator_profile_builder requires history, explicit_profile, llm_output, "
            "or a Bilibili creator_id/platform or creator_url"
        )

    if llm_output:
        profile = parse_creator_profile_response(
            llm_output,
            history=history,
            explicit_profile=explicit,
            platform=platform,
            creator_id=creator_id,
        )
    try:
        profile = generate_creator_profile_with_llm(
            history=history,
            explicit_profile=explicit,
            platform=platform,
            creator_id=creator_id,
        )
    except Exception:
        if not bool(input.get("allow_rule_fallback", False)):
            raise
        fallback = build_creator_profile_fallback(
            history=history,
            explicit_profile=explicit,
            platform=platform,
            creator_id=creator_id,
        )
        fallback["generation"] = {"method": "rule_fallback", "llm_failed": True}
        profile = fallback
    if collection_info:
        profile["data_collection"] = collection_info
    return profile


def normalize_creator_history(rows: Any) -> list[Dict[str, Any]]:
    """把 B 站采集结果转换为画像提示词使用的统一作品元数据。"""

    normalized: list[Dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not str(row.get("title") or "").strip():
            continue
        item: Dict[str, Any] = {
            "title": str(row.get("title") or "").strip(),
            "published_at": row.get("published_at"),
            "url": row.get("url"),
            "bvid": row.get("bvid"),
        }
        duration_seconds = _parse_duration_seconds(row.get("duration_seconds") or row.get("duration"))
        if duration_seconds is not None:
            item["duration_seconds"] = duration_seconds
        view_count = _parse_count(row.get("view_count") or row.get("view_count_text"))
        if view_count is not None:
            item["view_count"] = view_count
        if row.get("description"):
            item["description"] = str(row["description"])
        if isinstance(row.get("tags"), list) and row["tags"]:
            item["tags"] = [str(tag).strip() for tag in row["tags"] if str(tag).strip()]
        normalized.append({key: value for key, value in item.items() if value not in (None, "", [])})
    return normalized[:50]


def _parse_duration_seconds(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    if not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    return None


def _parse_count(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace(",", "")
    units = {"万": 10_000, "亿": 100_000_000, "千": 1_000, "K": 1_000, "k": 1_000, "M": 1_000_000, "m": 1_000_000}
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            try:
                return int(float(text[:-1]) * multiplier)
            except ValueError:
                return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    return int(float(match.group(0))) if match else None


def fetch_hot_news_source(input: Dict[str, Any], _ctx: ToolContext) -> Dict[str, Any]:
    """按 candidates/ranking 阶段执行受限数据库查询。"""

    return fetch_hot_news(
        stage=input.get("stage", "candidates"),
        start_time=input.get("start_time"),
        end_time=input.get("end_time"),
        platforms=input.get("platforms") or [],
        tag_ids=input.get("tag_ids") or [],
        candidate_ids=input.get("candidate_ids") or [],
        limit=min(int(input.get("limit") or 50), 50),
        base_url=input.get("base_url"),
    )


def score_proposal_topic(input: Dict[str, Any], _ctx: ToolContext) -> Dict[str, Any]:
    """proposal_v2 批量语义匹配评分；兼容单事件输入。"""

    events = [event for event in input.get("event_clusters") or [] if isinstance(event, dict)][:50]
    if not events and isinstance(input.get("event_cluster"), dict):
        events = [input["event_cluster"]]
    profile = input.get("creator_profile") or {}
    if not events or not profile:
        raise ValueError("proposal_topic_value_scorer requires event_cluster(s) and creator_profile")
    semantic_results = batch_semantic_fit_score(events, profile)
    scores = []
    for index, event in enumerate(events):
        event_id = str(event.get("event_id") or f"event_{index}")
        scores.append(
            score_event(
                event=event,
                profile=profile,
                search_records=input.get("search_records") or [],
                candidate_angles=input.get("candidate_angles") or [],
                semantic_result=semantic_results[event_id],
            )
        )
    if input.get("event_clusters"):
        return {"scoring_version": "proposal_v2", "scores": scores}
    return scores[0]


def plan_angles(input: Dict[str, Any], _ctx: ToolContext) -> Dict[str, Any]:
    return plan_content_angles(
        event=input.get("event_cluster") or {},
        profile=input.get("creator_profile") or {},
        count=min(int(input.get("count") or 5), 5),
    )


def check_event_risk_evidence(input: Dict[str, Any], _ctx: ToolContext) -> Dict[str, Any]:
    return check_risk_evidence(
        event=input.get("event_cluster") or {},
        search_records=input.get("search_records") or [],
    )


def recommend_creator_topics(input: Dict[str, Any], _ctx: ToolContext) -> Dict[str, Any]:
    """连接画像、事件和证据的 proposal_v1 排序入口。"""

    profile = input.get("creator_profile") or {}
    events = [event for event in input.get("events") or [] if isinstance(event, dict)][:50]
    if not profile:
        raise ValueError("creator_topic_recommender requires an LLM-parsed creator_profile")
    scored = []
    search_records = input.get("search_records") or []
    for event in events:
        score = score_event(event, profile, search_records)
        scored.append({
            "event": event,
            "score": score,
            "risk_evidence": check_risk_evidence(event, search_records),
            "content_angles": plan_content_angles(event, profile, 5),
        })
    scored.sort(key=lambda item: item["score"]["topic_value"], reverse=True)
    limit = max(1, min(int(input.get("limit") or 10), 50))
    return {
        "scoring_version": "proposal_v1",
        "creator_profile_id": profile.get("creator_id"),
        "recommendations": scored[:limit],
        "total_events": len(events),
    }


HANDLER_BY_TOOL_NAME = {
    "creator_profile_builder": build_creator_profile,
    "hot_news_fetcher": fetch_hot_news_source,
    "proposal_topic_value_scorer": score_proposal_topic,
    "content_angle_planner": plan_angles,
    "risk_evidence_checker": check_event_risk_evidence,
    "creator_topic_recommender": recommend_creator_topics,
    "web_search_fetcher": fetch_web_search,
    "source_bundle_normalizer": normalize_source_bundle,
}
