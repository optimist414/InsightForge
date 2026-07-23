"""创作者画像的 LLM 提示词、结构化输出协议和解析器。

画像生成是一个独立的结构化阶段：模型只根据提供的证据归纳，不负责访问
数据库或联网；取数由其它工具完成，解析器负责把模型输出变成下游可消费的对象。

项目依赖：由 `handlers.py` 调用，并使用环境变量中的 LLM 配置。
外部依赖：DeepSeek 兼容 Chat Completions API；网络请求使用 Python 标准库 urllib。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


CREATOR_PROFILE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "creator_identity_tags",
        "content_positioning",
        "content_format",
        "creator_size",
        "domain_profile",
        "audience_profile",
        "evidence_summary",
        "confidence",
    ],
    "properties": {
        "creator_identity_tags": {
            "type": "array",
            "description": "可用于筛选选题的短标签，例如科技区、教程型、个人创作者。",
            "items": {"type": "string"},
            "maxItems": 12,
        },
        "content_positioning": {
            "type": "object",
            "additionalProperties": False,
            "required": ["one_sentence", "value_proposition", "style"],
            "properties": {
                "one_sentence": {"type": "string"},
                "value_proposition": {"type": "string"},
                "style": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            },
        },
        "content_format": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primary", "formats", "duration_range_seconds"],
            "properties": {
                "primary": {"type": "string", "enum": ["short_video", "long_video", "mixed", "unknown"]},
                "formats": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "duration_range_seconds": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["min", "max", "median"],
                    "properties": {
                        "min": {"type": ["integer", "null"], "minimum": 0},
                        "max": {"type": ["integer", "null"], "minimum": 0},
                        "median": {"type": ["integer", "null"], "minimum": 0},
                    },
                },
            },
        },
        "creator_size": {
            "type": "object",
            "additionalProperties": False,
            "required": ["level", "follower_count", "median_views"],
            "properties": {
                "level": {"type": "string", "enum": ["top", "large", "medium", "small", "nano", "unknown"]},
                "follower_count": {"type": ["integer", "null"], "minimum": 0},
                "median_views": {"type": ["integer", "null"], "minimum": 0},
            },
        },
        "domain_profile": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primary", "secondary", "main_tags", "avoid_tags"],
            "properties": {
                "primary": {"type": "string"},
                "secondary": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "main_tags": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
                "avoid_tags": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
            },
        },
        "audience_profile": {
            "type": "object",
            "additionalProperties": False,
            "required": ["description", "needs", "knowledge_level"],
            "properties": {
                "description": {"type": "string"},
                "needs": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                "knowledge_level": {"type": "string"},
            },
        },
        "evidence_summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sample_count", "supported_claims", "unknowns"],
            "properties": {
                "sample_count": {"type": "integer", "minimum": 0},
                "supported_claims": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "unknowns": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}


PROFILE_SYSTEM_PROMPT = """你是创作者画像归纳器，不是选题推荐器。
你只能依据用户提供的创作者资料和作品样本归纳画像，不得联网、不得补造粉丝数、播放量、受众或身份。
证据不足时必须写 unknown 或放入 evidence_summary.unknowns。
只输出一个 JSON 对象，不要 Markdown 代码块，不要解释，不要输出 JSON 以外的文字。
必须严格遵循给定的 JSON Schema。画像中的标签要短、稳定、可用于后续选题匹配；不要把单个视频标题直接当成创作者身份。
""".strip()


def build_creator_profile_prompt(
    history: Optional[List[Dict[str, Any]]] = None,
    explicit_profile: Optional[Dict[str, Any]] = None,
    platform: str = "",
    creator_id: str = "",
) -> Dict[str, Any]:
    """构造 profile 阶段的 Chat Completions 输入。

    只发送最近最多 50 条作品样本，避免历史数据无限膨胀；作品字段也只保留
    画像需要的元数据。
    """

    compact_history = []
    # 约定 history 按最新到最旧传入；只取头部，避免把较早作品误当成最近样本。
    for row in (history or [])[:50]:
        if not isinstance(row, dict):
            continue
        compact_history.append(
            {
                key: row.get(key)
                for key in (
                    "title", "description", "tags", "topic_tags", "published_at",
                    "duration", "duration_seconds", "view_count", "views", "like_count",
                    "platform", "url",
                )
                if row.get(key) is not None
            }
        )
    context = {
        "creator_id": creator_id or (explicit_profile or {}).get("creator_id"),
        "platform": platform or (explicit_profile or {}).get("platform"),
        "explicit_profile": explicit_profile or {},
        "recent_work_samples": compact_history,
    }
    system_prompt = (
        PROFILE_SYSTEM_PROMPT
        + "\n\n输出 JSON Schema（字段名和枚举必须遵守）：\n"
        + json.dumps(CREATOR_PROFILE_OUTPUT_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    )
    return {
        "system": system_prompt,
        "messages": [{"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        "response_format": {"type": "json_object"},
        "output_schema": CREATOR_PROFILE_OUTPUT_SCHEMA,
        "sample_count": len(compact_history),
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("profile LLM output does not contain a JSON object")
        value, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("profile LLM output must be a JSON object")
    return value


def _string_list(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))[:limit]


def _non_negative_int(value: Any) -> Optional[int]:
    if value is None or value == "" or str(value).lower() == "unknown":
        return None


def _validate_profile_shape(raw: Dict[str, Any]) -> None:
    """校验模型是否遵守画像协议；字段缺失时不静默生成结论。"""

    required = set(CREATOR_PROFILE_OUTPUT_SCHEMA["required"])
    missing = sorted(key for key in required if key not in raw)
    if missing:
        raise ValueError("profile JSON missing required fields: " + ", ".join(missing))
    nested_required = {
        "content_positioning": ["one_sentence", "value_proposition", "style"],
        "content_format": ["primary", "formats", "duration_range_seconds"],
        "creator_size": ["level", "follower_count", "median_views"],
        "domain_profile": ["primary", "secondary", "main_tags", "avoid_tags"],
        "audience_profile": ["description", "needs", "knowledge_level"],
        "evidence_summary": ["sample_count", "supported_claims", "unknowns"],
    }
    for object_name, keys in nested_required.items():
        value = raw.get(object_name)
        if not isinstance(value, dict):
            raise ValueError("profile field must be an object: " + object_name)
        missing_nested = [key for key in keys if key not in value]
        if missing_nested:
            raise ValueError(
                "profile object {} missing: {}".format(object_name, ", ".join(missing_nested))
            )
    duration = raw["content_format"]["duration_range_seconds"]
    if not isinstance(duration, dict) or any(key not in duration for key in ("min", "max", "median")):
        raise ValueError("content_format.duration_range_seconds must contain min, max and median")
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return None


def parse_creator_profile_response(
    text: str,
    history: Optional[List[Dict[str, Any]]] = None,
    explicit_profile: Optional[Dict[str, Any]] = None,
    platform: str = "",
    creator_id: str = "",
) -> Dict[str, Any]:
    """解析并规范化 LLM 画像，输出字段稳定地供指标计算使用。"""

    raw = _extract_json_object(text)
    _validate_profile_shape(raw)
    positioning = raw.get("content_positioning") or {}
    formats = raw.get("content_format") or {}
    duration = formats.get("duration_range_seconds") or {}
    size = raw.get("creator_size") or {}
    domains = raw.get("domain_profile") or {}
    audience = raw.get("audience_profile") or {}
    evidence = raw.get("evidence_summary") or {}
    sample_count = _non_negative_int(evidence.get("sample_count"))
    if sample_count is None:
        sample_count = min(len(history or []), 50)
    primary_domain = str(domains.get("primary") or "unknown").strip() or "unknown"
    primary_format = str(formats.get("primary") or "unknown").strip() or "unknown"
    profile = {
        "creator_id": creator_id or (explicit_profile or {}).get("creator_id"),
        "creator_name": (explicit_profile or {}).get("creator_name"),
        "platform": platform or (explicit_profile or {}).get("platform"),
        "domain": primary_domain,
        "domains": [{"name": primary_domain, "score": 1.0, "evidence_titles": []}],
        "main_tags": _string_list(domains.get("main_tags"), 15),
        "avoid_tags": _string_list(domains.get("avoid_tags"), 10),
        "identity_tags": _string_list(raw.get("creator_identity_tags"), 12),
        "content_positioning": {
            "one_sentence": str(positioning.get("one_sentence") or "unknown"),
            "value_proposition": str(positioning.get("value_proposition") or "unknown"),
            "audience": str(audience.get("description") or "unknown"),
            "style": _string_list(positioning.get("style"), 6),
        },
        "content_format": {
            "primary": primary_format,
            "formats": _string_list(formats.get("formats"), 6),
            "sample_count": sample_count,
            "duration_seconds_median": _non_negative_int(duration.get("median")),
            "duration_range_seconds": {
                "min": _non_negative_int(duration.get("min")),
                "max": _non_negative_int(duration.get("max")),
            },
            "evidence": "llm_structured_from_recent_work_samples",
        },
        "creator_size": {
            "level": str(size.get("level") or "unknown"),
            "follower_count": _non_negative_int(size.get("follower_count")),
            "median_views": _non_negative_int(size.get("median_views")),
            "evidence": "llm_structured_from_explicit_stats" if size.get("follower_count") is not None else "unknown_if_not_provided",
        },
        "target_platforms": [platform] if platform else [],
        "risk_preference": (explicit_profile or {}).get("risk_preference") or "normal",
        "audience_profile": {
            "description": str(audience.get("description") or "unknown"),
            "needs": _string_list(audience.get("needs"), 8),
            "knowledge_level": str(audience.get("knowledge_level") or "unknown"),
        },
        "history_summary": {
            "sample_count": sample_count,
            "source": "llm_profile_generation",
            "confidence": str(raw.get("confidence") or "low"),
            "supported_claims": _string_list(evidence.get("supported_claims"), 12),
            "unknowns": _string_list(evidence.get("unknowns"), 12),
        },
        "generation": {
            "method": "llm_json_profile_v1",
            "raw_output_preserved": False,
        },
        "downstream_usage": {
            "fit_matching": ["domain", "main_tags", "avoid_tags"],
            "format_matching": ["content_format", "target_platforms"],
            "scale_context": ["creator_size"],
        },
    }
    if not profile["identity_tags"]:
        profile["identity_tags"] = ["领域:" + primary_domain, "形式:" + primary_format]
    return profile


def generate_creator_profile_with_llm(
    history: Optional[List[Dict[str, Any]]] = None,
    explicit_profile: Optional[Dict[str, Any]] = None,
    platform: str = "",
    creator_id: str = "",
    llm_client: Any = None,
) -> Dict[str, Any]:
    """通过无工具的独立 Chat Completions 调用生成并解析画像。"""

    if llm_client is None:
        from ...engine.sse_client import ChatCompletionsSseClient

        llm_client = ChatCompletionsSseClient()
    request = build_creator_profile_prompt(history, explicit_profile, platform, creator_id)
    stream_request = {
        "messages": [
            {"role": "system", "content": request["system"]},
            request["messages"][0],
        ],
        "response_format": request["response_format"],
        "stream": True,
    }
    from ...engine.core import StreamEventType

    chunks: List[str] = []
    errors: List[str] = []
    for event in llm_client.create_message_stream(stream_request):
        if event.event_type == StreamEventType.CONTENT_BLOCK_DELTA and event.block_type == "text":
            chunks.append(event.delta or "")
        elif event.event_type == StreamEventType.ERROR:
            errors.append(event.error or "unknown LLM error")
    if errors:
        raise RuntimeError("; ".join(errors))
    return parse_creator_profile_response(
        "".join(chunks),
        history=history,
        explicit_profile=explicit_profile,
        platform=platform,
        creator_id=creator_id,
    )
