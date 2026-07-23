"""proposal_v1 选题指标与内容规划。

作用：根据事件、搜索证据和创作者画像计算选题价值，生成内容角度及风险证据提示。
项目依赖：`creator_profile.py` 提供领域关键词；由 `handlers.py` 调用。
外部依赖：仅使用 Python 标准库，不直接访问数据库或网络。
"""

from __future__ import annotations

import re
import json
from statistics import mean
from typing import Any, Dict, List

from .creator_profile import DOMAIN_KEYWORDS


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _items(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = event.get("items") or event.get("hot_news_items") or []
    return [item for item in items if isinstance(item, dict)] or [event]


def _latest_rank(item: Dict[str, Any]) -> int | None:
    if item.get("rank_no") is not None:
        return int(item["rank_no"])
    ranks = [record for record in item.get("rank_records") or [] if record.get("rank_no") is not None]
    if not ranks:
        return None
    ranks.sort(key=lambda record: str(record.get("record_time") or ""))
    return int(ranks[-1]["rank_no"])


def _rank_score(rank: int | None) -> float:
    if rank is None:
        return 45.0
    return _clamp((51 - rank) / 50 * 100)


def hot_score(event: Dict[str, Any]) -> Dict[str, Any]:
    items = _items(event)
    rank_scores = [_rank_score(_latest_rank(item)) for item in items]
    platforms = {str(item.get("platform_code") or item.get("platform")) for item in items if item.get("platform_code") or item.get("platform")}
    platform_score = min(len(platforms), 3) / 3 * 100
    value = 0.8 * (mean(rank_scores) if rank_scores else 45.0) + 0.2 * platform_score
    return {
        "hot_score": round(value, 2),
        "average_rank_score": round(mean(rank_scores) if rank_scores else 45.0, 2),
        "platform_coverage_score": round(platform_score, 2),
        "platform_count": len(platforms),
        "evidence": "latest rank and cross-platform occurrence",
    }


def rise_score(event: Dict[str, Any]) -> Dict[str, Any]:
    deltas: List[float] = []
    for item in _items(event):
        records = [record for record in item.get("rank_records") or [] if record.get("rank_no") is not None]
        records.sort(key=lambda record: str(record.get("record_time") or ""))
        if len(records) >= 2:
            deltas.append(float(records[-2]["rank_no"]) - float(records[-1]["rank_no"]))
    if deltas:
        value = _clamp(50 + mean(deltas) * 2)
        reason = "compared latest two rank records"
    else:
        value = 70.0 if any(item.get("first_seen_time") and not item.get("rank_records") for item in _items(event)) else 50.0
        reason = "new-event default" if value == 70.0 else "no rank history"
    return {"rise_score": round(value, 2), "average_rank_delta": round(mean(deltas), 2) if deltas else None, "evidence": reason}


def _event_terms(event: Dict[str, Any]) -> set[str]:
    text = " ".join(
        [str(event.get("representative_title") or event.get("title") or "")] + [str(tag) for tag in event.get("tags") or []]
    ).lower()
    return {term for term in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z][a-z0-9+#.-]{1,}", text)}


SEMANTIC_FIT_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["event_id", "semantic_fit", "matched_themes", "rationale"],
                "properties": {
                    "event_id": {"type": "string"},
                    "semantic_fit": {"type": "number", "minimum": 0, "maximum": 100},
                    "matched_themes": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                    "rationale": {"type": "string", "maxLength": 500},
                },
            },
        }
    },
}


def build_semantic_fit_prompt(events: List[Dict[str, Any]], profile: Dict[str, Any]) -> Dict[str, Any]:
    """构造批量语义匹配请求；只服务于 proposal_topic_value_scorer。"""

    payload_events = []
    for index, event in enumerate(events):
        payload_events.append(
            {
                "event_id": str(event.get("event_id") or f"event_{index}"),
                "title": event.get("representative_title") or event.get("title") or "",
                "tags": event.get("tags") or [],
            }
        )
    creator = {
        "domain": profile.get("domain"),
        "domains": profile.get("domains") or [],
        "main_tags": profile.get("main_tags") or [],
        "avoid_tags": profile.get("avoid_tags") or [],
        "content_positioning": profile.get("content_positioning") or {},
    }
    system = (
        "你是选题适配度评估器，只评估热点事件与创作者画像的语义相关性。"
        "不要判断热度、真实性、风险或是否值得发布，不要补造事件事实。"
        "semantic_fit 为 0 到 100；只能从创作者已有 domain、domains、main_tags 和 content_positioning 中归纳 matched_themes。"
        "必须为每个 event_id 返回一个结果，只输出符合 JSON Schema 的 JSON 对象（json object）。\n"
        + json.dumps(SEMANTIC_FIT_OUTPUT_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"creator_profile": creator, "events": payload_events}, ensure_ascii=False)},
        ],
    }


def _parse_semantic_fit_response(text: str, events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(text)
    rows = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError("semantic fit response must contain results array")
    valid_ids = {str(event.get("event_id") or f"event_{index}") for index, event in enumerate(events)}
    result = {}
    for row in rows:
        if not isinstance(row, dict) or str(row.get("event_id")) not in valid_ids:
            continue
        score = _clamp(float(row.get("semantic_fit", 0)))
        result[str(row["event_id"])] = {
            "semantic_fit": round(score, 2),
            "matched_themes": [str(item).strip() for item in row.get("matched_themes") or [] if str(item).strip()][:10],
            "rationale": str(row.get("rationale") or "").strip()[:500],
            "evidence": "batch LLM semantic fit evaluation",
        }
    missing = valid_ids - result.keys()
    if missing:
        raise ValueError("semantic fit response missing event ids: " + ", ".join(sorted(missing)))
    return result


def batch_semantic_fit_score(events: List[Dict[str, Any]], profile: Dict[str, Any], llm_client: Any = None) -> Dict[str, Dict[str, Any]]:
    """一次调用 LLM，为一批事件生成语义匹配结果。"""

    if llm_client is None:
        from ...engine.sse_client import ChatCompletionsSseClient
        llm_client = ChatCompletionsSseClient()
    from ...engine.core import StreamEventType

    request = build_semantic_fit_prompt(events, profile)
    chunks: List[str] = []
    errors: List[str] = []
    for event in llm_client.create_message_stream({**request, "stream": True}):
        if event.event_type == StreamEventType.CONTENT_BLOCK_DELTA and event.block_type == "text":
            chunks.append(event.delta)
        elif event.event_type == StreamEventType.ERROR:
            errors.append(event.error)
    if errors:
        raise RuntimeError("semantic fit LLM failed: " + "; ".join(errors))
    return _parse_semantic_fit_response("".join(chunks), events)


def _legacy_fit_score(event: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """兼容未走 proposal_topic_value_scorer 的旧内部调用。"""

    terms = _event_terms(event)
    creator_terms = {str(item).lower() for item in profile.get("main_tags") or []}
    creator_terms.update(str(item).lower() for item in profile.get("domains") or [] if not isinstance(item, dict))
    domain = str(profile.get("domain") or "").lower()
    if domain:
        creator_terms.add(domain)
    matched = sorted(term for term in terms if term in creator_terms or any(term in creator for creator in creator_terms))
    avoid = {str(item).lower() for item in profile.get("avoid_tags") or []}
    avoided = sorted(term for term in terms if term in avoid or any(term in item for item in avoid))
    strong_domain = bool(domain and (domain in terms or any(keyword in terms for keyword in DOMAIN_KEYWORDS.get(profile.get("domain"), []))))
    base = 60.0 if strong_domain else 35.0 if matched else 20.0
    value = _clamp(base + min(len(set(matched)) * 10, 40) - (30 if avoided else 0))
    return {
        "fit_score": round(value, 2),
        "base_domain_score": base,
        "matched_keywords": matched[:10],
        "avoid_matches": avoided[:10],
        "evidence": "legacy lexical fit for non-scorer internal call",
    }


def fit_score(event: Dict[str, Any], profile: Dict[str, Any], semantic_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """使用批量 LLM 语义结果计算创作者匹配分，替代旧的字面关键词匹配。"""

    if semantic_result is None:
        raise ValueError("semantic fit result is required for proposal_topic_value_scorer")
    value = _clamp(float(semantic_result.get("semantic_fit", 0)))
    return {
        "fit_score": round(value, 2),
        "semantic_fit": round(value, 2),
        "matched_themes": semantic_result.get("matched_themes") or [],
        "fit_rationale": semantic_result.get("rationale") or "",
        "evidence": semantic_result.get("evidence") or "batch LLM semantic fit evaluation",
    }


def _risk(event: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join([str(event.get("representative_title") or event.get("title") or "")] + [str(tag) for tag in event.get("tags") or []])
    rules = {
        "rumor": ["网传", "爆料", "疑似", "传闻", "据称"],
        "finance": ["股票", "荐股", "买入", "卖出", "暴涨", "暴跌", "基金"],
        "medical": ["医疗", "诊断", "处方", "治疗", "癌症"],
        "legal": ["判决", "起诉", "犯罪", "拘留", "违法"],
        "privacy": ["身份证", "住址", "电话", "隐私"],
        "disaster": ["死亡", "灾难", "伤亡", "坠毁", "地震"],
    }
    flags = [name for name, words in rules.items() if any(word in text for word in words)]
    if "disaster" in flags or len(flags) >= 3:
        level, factor = "block", 0.0
    elif "finance" in flags or "medical" in flags or "rumor" in flags or len(flags) >= 2:
        level, factor = "high", 0.5
    elif flags:
        level, factor = "medium", 0.8
    else:
        level, factor = "low", 1.0
    return {"risk_level": level, "risk_factor": factor, "risk_flags": flags}


def create_score(event: Dict[str, Any], profile: Dict[str, Any], search_records: List[Dict[str, Any]] | None = None, candidate_angles: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    title = str(event.get("representative_title") or event.get("title") or "该热点")
    angles = candidate_angles or [
        {"angle_type": "信息增量", "angle": f"用时间线讲清楚“{title}”发生了什么"},
        {"angle_type": "知识解释", "angle": f"解释“{title}”背后的概念、机制或行业影响"},
        {"angle_type": "普通人影响", "angle": f"回答“{title}”和目标受众有什么关系"},
    ]
    valid_angles = [angle for angle in angles if isinstance(angle, dict) and str(angle.get("angle") or "").strip()]
    angle_score = min(len(valid_angles), 5) / 5 * 100
    source_count = len(event.get("items") or []) + len(search_records or [])
    evidence_score = min(source_count, 3) / 3 * 100
    platforms = set(str(item.get("platform_code") or item.get("platform")) for item in _items(event))
    target_platforms = set(str(item) for item in profile.get("target_platforms") or [])
    format_score = 100.0 if platforms & target_platforms else 60.0 if target_platforms else 60.0
    value = 0.4 * angle_score + 0.3 * evidence_score + 0.3 * format_score
    return {
        "create_score": round(value, 2),
        "angle_count_score": round(angle_score, 2),
        "evidence_sufficiency_score": round(evidence_score, 2),
        "format_fit_score": round(format_score, 2),
        "suggested_angles": valid_angles[:5],
    }


def score_event(event: Dict[str, Any], profile: Dict[str, Any] | None = None, search_records: List[Dict[str, Any]] | None = None, candidate_angles: List[Dict[str, Any]] | None = None, semantic_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
    profile = profile or {}
    hot = hot_score(event)
    rise = rise_score(event)
    fit = fit_score(event, profile, semantic_result) if semantic_result is not None else _legacy_fit_score(event, profile)
    create = create_score(event, profile, search_records, candidate_angles)
    risk = _risk(event)
    raw = 0.25 * hot["hot_score"] + 0.20 * rise["rise_score"] + 0.35 * fit["fit_score"] + 0.20 * create["create_score"]
    topic_value = round(raw * risk["risk_factor"], 2)
    level = "强推荐" if topic_value >= 80 else "可推荐" if topic_value >= 65 else "观察" if topic_value >= 50 else "不推荐"
    return {
        "topic_value": topic_value,
        "recommendation_level": level,
        "scores": {**hot, **rise, **fit, **create, **risk, "raw_score": round(raw, 2)},
        "explanation": [
            f"热度分 {hot['hot_score']}，覆盖 {hot['platform_count']} 个平台",
            f"上升分 {rise['rise_score']}，依据：{rise['evidence']}",
            f"语义匹配分 {fit['fit_score']}，命中主题：{', '.join(fit.get('matched_themes', fit.get('matched_keywords', []))) or '无'}",
            f"可创作分 {create['create_score']}，有效角度 {len(create['suggested_angles'])} 个",
            f"风险等级 {risk['risk_level']}，折扣 {risk['risk_factor']}",
        ],
    }


def plan_content_angles(
    event: Dict[str, Any],
    profile: Dict[str, Any] | None = None,
    count: int = 5,
) -> Dict[str, Any]:
    """生成可执行但不冒充事实的内容角度。"""

    profile = profile or {}
    title = str(event.get("representative_title") or event.get("title") or "该热点")
    primary = str(profile.get("content_format", {}).get("primary") or "unknown")
    angle_pool = [
        {"angle_type": "时间线", "angle": f"把“{title}”按时间线讲清楚，区分已确认事实和待核实信息"},
        {"angle_type": "机制解释", "angle": f"解释“{title}”背后的机制、行业背景或关键概念"},
        {"angle_type": "普通人影响", "angle": f"回答“{title}”对普通用户、消费者或从业者意味着什么"},
        {"angle_type": "对比分析", "angle": f"围绕“{title}”比较变化前后、不同方案或不同观点"},
        {"angle_type": "行动清单", "angle": f"把“{title}”转化为受众可以执行的判断或行动清单"},
    ]
    safe_count = max(1, min(int(count or 5), 5))
    return {
        "event_title": title,
        "angles": angle_pool[:safe_count],
        "recommended_format": primary,
        "is_fact_claim": False,
    }


def check_risk_evidence(
    event: Dict[str, Any],
    search_records: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """把风险规则和证据数量检查分开，便于最终答案说明原因。"""

    risk = _risk(event)
    event_sources = len(_items(event))
    external_sources = len(search_records or [])
    total_sources = event_sources + external_sources
    needs = []
    if total_sources < 2:
        needs.append("至少补充一个独立来源")
    if risk["risk_level"] in {"high", "block"}:
        needs.append("补充权威原始来源并进行人工复核")
    return {
        **risk,
        "event_source_count": event_sources,
        "external_source_count": external_sources,
        "evidence_sufficient": total_sources >= 2 and risk["risk_level"] not in {"block"},
        "evidence_needs": needs,
    }
