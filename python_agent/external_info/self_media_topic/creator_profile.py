"""创作者画像规则兜底模块。

作用：在画像 LLM 不可用且显式允许兜底时，根据作品标题、时长和互动数据生成基础画像。
项目依赖：无本项目模块依赖；由 `handlers.py` 调用。
外部依赖：仅使用 Python 标准库。
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


DOMAIN_KEYWORDS = {
    "人工智能": ["ai", "人工智能", "大模型", "agent", "智能体", "机器学习", "深度学习"],
    "科技数码": ["科技", "数码", "手机", "电脑", "芯片", "硬件", "软件", "互联网", "app"],
    "编程开发": ["java", "python", "javascript", "后端", "前端", "编程", "api", "开源", "开发"],
    "财经商业": ["股票", "基金", "投资", "财经", "商业", "公司", "创业", "经济"],
    "教育职场": ["教育", "学习", "考试", "求职", "职场", "职业", "面试", "大学"],
    "游戏": ["游戏", "steam", "主机", "电竞", "galgame", "手游"],
    "生活情感": ["生活", "情感", "恋爱", "婚姻", "相亲", "旅行", "美食", "家居"],
    "影视娱乐": ["电影", "电视剧", "影视", "明星", "娱乐", "综艺", "动漫"],
}

FORMAT_KEYWORDS = {
    "教程/讲解": ["教程", "入门", "怎么", "如何", "详解", "原理", "指南", "攻略"],
    "测评/对比": ["测评", "评测", "体验", "对比", "开箱", "值得买吗", "推荐"],
    "观点/评论": ["为什么", "怎么看", "观点", "评论", "真相", "影响", "变化"],
    "故事/记录": ["记录", "日常", "故事", "经历", "挑战", "旅行", "相亲"],
    "盘点/清单": ["盘点", "合集", "清单", "排名", "推荐", "必看", "top"],
}


def parse_duration_seconds(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = str(value).strip()
    parts = text.split(":")
    if not all(part.isdigit() for part in parts):
        return None
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return None


def _text_from_row(row: Dict[str, Any]) -> str:
    values = [row.get("title"), row.get("description"), row.get("summary"), row.get("tags")]
    return " ".join(str(value) for value in values if value)


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9+#.-]{1,}", text)]


def _top_domains(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    domain_scores: Counter[str] = Counter()
    evidence: Dict[str, List[str]] = {}
    for row in rows:
        text = _text_from_row(row).lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            hits = [keyword for keyword in keywords if keyword.lower() in text]
            if hits:
                domain_scores[domain] += min(len(hits), 3)
                evidence.setdefault(domain, []).append(str(row.get("title") or ""))
    total = sum(domain_scores.values()) or 1
    return [
        {
            "name": domain,
            "score": round(score / total, 4),
            "evidence_titles": [title for title in evidence.get(domain, []) if title][:3],
        }
        for domain, score in domain_scores.most_common(5)
    ]


def _top_tags(rows: List[Dict[str, Any]], domains: List[Dict[str, Any]]) -> List[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        tags = row.get("topic_tags") or row.get("tags") or []
        if isinstance(tags, str):
            tags = re.split(r"[,，/、\s]+", tags)
        for tag in tags or []:
            tag_text = str(tag).strip()
            if tag_text:
                counter[tag_text] += 2
        for token in _tokens(_text_from_row(row)):
            if len(token) >= 2:
                counter[token] += 1
    tags = [tag for tag, _ in counter.most_common(10)]
    tags.extend(domain["name"] for domain in domains)
    return list(dict.fromkeys(tags))[:12]


def _format_profile(rows: List[Dict[str, Any]], explicit: Dict[str, Any]) -> Dict[str, Any]:
    durations = [
        duration
        for duration in (parse_duration_seconds(row.get("duration") or row.get("duration_seconds")) for row in rows)
        if duration is not None
    ]
    short_count = sum(duration <= 180 for duration in durations)
    long_count = sum(duration >= 300 for duration in durations)
    if explicit.get("content_formats"):
        formats = [str(item) for item in explicit["content_formats"]]
        primary = "mixed" if len(formats) > 1 else formats[0]
    elif not durations:
        primary = "unknown"
        formats = []
    elif short_count / len(durations) >= 0.65:
        primary, formats = "short_video", ["短视频"]
    elif long_count / len(durations) >= 0.65:
        primary, formats = "long_video", ["长视频"]
    else:
        primary, formats = "mixed", ["短视频", "长视频"]
    return {
        "primary": primary,
        "formats": formats,
        "sample_count": len(rows),
        "duration_seconds_median": int(median(durations)) if durations else None,
        "evidence": "explicit_profile" if explicit.get("content_formats") else "history_duration" if durations else "insufficient_data",
    }


def _size_profile(rows: List[Dict[str, Any]], explicit: Dict[str, Any]) -> Dict[str, Any]:
    follower_count = explicit.get("follower_count") or explicit.get("followers")
    if follower_count is not None:
        followers = max(0, int(float(follower_count)))
        if followers >= 10_000_000:
            level = "top"
        elif followers >= 1_000_000:
            level = "large"
        elif followers >= 100_000:
            level = "medium"
        elif followers >= 10_000:
            level = "small"
        else:
            level = "nano"
        return {"level": level, "follower_count": followers, "evidence": "explicit_or_platform_stats"}
    views = [float(row.get("view_count") or row.get("views") or 0) for row in rows]
    views = [view for view in views if view > 0]
    return {
        "level": "unknown",
        "follower_count": None,
        "median_views": int(median(views)) if views else None,
        "evidence": "followers_missing_do_not_infer_creator_size",
    }


def build_creator_profile(
    history: Optional[List[Dict[str, Any]]] = None,
    explicit_profile: Optional[Dict[str, Any]] = None,
    platform: str = "",
    creator_id: str = "",
) -> Dict[str, Any]:
    """从历史作品和显式账号信息构造可供后续匹配的创作者画像。"""

    rows = [row for row in history or [] if isinstance(row, dict) and row.get("title")]
    explicit = explicit_profile or {}
    domains = _top_domains(rows)
    main_tags = list(dict.fromkeys([str(tag) for tag in explicit.get("main_tags") or []] + _top_tags(rows, domains)))[:15]
    primary_domain = str(explicit.get("domain") or (domains[0]["name"] if domains else "未确定"))
    formats = _format_profile(rows, explicit)
    size = _size_profile(rows, explicit)
    identity_tags = [
        f"领域:{primary_domain}",
        f"形式:{formats['primary']}",
        f"体量:{size['level']}",
    ]
    if platform:
        identity_tags.append(f"平台:{platform}")
    identity_tags.extend(f"主题:{tag}" for tag in main_tags[:5])
    return {
        "creator_id": creator_id or explicit.get("creator_id"),
        "creator_name": explicit.get("creator_name"),
        "platform": platform or explicit.get("platform"),
        "domain": primary_domain,
        "domains": domains,
        "main_tags": main_tags,
        "avoid_tags": [str(tag) for tag in explicit.get("avoid_tags") or []],
        "identity_tags": list(dict.fromkeys(identity_tags)),
        "content_positioning": {
            "value_proposition": explicit.get("value_proposition") or f"围绕{primary_domain}提供可理解、可执行的信息和观点",
            "audience": explicit.get("audience") or "待补充",
            "style": explicit.get("style") or ["解释型内容"],
        },
        "content_format": formats,
        "creator_size": size,
        "target_platforms": explicit.get("target_platforms") or ([platform] if platform else []),
        "risk_preference": explicit.get("risk_preference") or "normal",
        "history_summary": {
            "sample_count": len(rows),
            "source": explicit.get("history_source") or "provided_history",
            "confidence": "medium" if len(rows) >= 10 else "low",
        },
        "downstream_usage": {
            "fit_matching": ["domain", "main_tags", "avoid_tags", "target_platforms"],
            "format_matching": ["content_format", "content_positioning.style"],
            "scale_context": ["creator_size"],
        },
    }
