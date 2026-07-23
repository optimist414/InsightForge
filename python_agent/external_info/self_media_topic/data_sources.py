"""自媒体选题业务的数据源适配。

作用：把热点候选召回和排名证据查询拆成两个受限阶段，供 hot_news_fetcher 使用。
项目依赖：`java_sql_proxy.py` 负责向 Java 后端转发生成的只读 SQL。
外部依赖：Java SQL 只读接口及其背后的 MySQL 数据库。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from ..java_sql_proxy import query_java_sql_proxy


MAX_CANDIDATES = 50
MAX_TAG_IDS = 3
MAX_RANKING_IDS = 20
MAX_RANKING_ROWS = 100


def _sql_literal(value: str) -> str:
    """为内部生成的 SQL 转义字符串；调用方不能注入 SQL 片段。"""

    return value.replace("\\", "\\\\").replace("'", "''")


def _time_text(value: Any, default: datetime) -> str:
    if value is None or str(value).strip() == "":
        value = default
    if isinstance(value, datetime):
        return value.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    return str(value).strip().replace("T", " ")[:19]


def _bounded_ints(values: Optional[Iterable[Any]], maximum: int) -> List[int]:
    result: List[int] = []
    for value in values or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in result:
            result.append(number)
        if len(result) >= maximum:
            break
    return result


def _time_window(start_time: Optional[Any], end_time: Optional[Any]) -> tuple[str, str]:
    end_default = datetime.now().replace(microsecond=0)
    start_default = end_default - timedelta(days=1)
    return (
        _time_text(start_time, start_default),
        _time_text(end_time, end_default),
    )


def build_hot_news_sql(
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None,
    platforms: Optional[Iterable[Any]] = None,
    tag_ids: Optional[Iterable[Any]] = None,
    limit: int = MAX_CANDIDATES,
) -> str:
    """生成第一阶段候选 SQL，只访问 hot_news 和可选的 hot_news_tag。"""

    start, end = _time_window(start_time, end_time)
    safe_limit = max(1, min(int(limit or MAX_CANDIDATES), MAX_CANDIDATES))
    platform_values = [str(item).strip() for item in platforms or [] if str(item).strip()][:20]
    platform_clause = ""
    if platform_values:
        quoted = ", ".join("'{}'".format(_sql_literal(item)) for item in platform_values)
        platform_clause = " AND hn.platform_code IN ({})".format(quoted)

    tag_values = _bounded_ints(tag_ids, MAX_TAG_IDS)
    tag_clause = ""
    join_clause = ""
    if tag_values:
        join_clause = " INNER JOIN hot_news_tag hnt ON hnt.hot_news_id = hn.id"
        tag_clause = " AND hnt.tag_id IN ({})".format(", ".join(str(item) for item in tag_values))

    return (
        "SELECT DISTINCT hn.id, hn.title, hn.url, hn.platform_code, hn.platform_name, "
        "hn.first_seen_time, hn.latest_seen_time "
        "FROM hot_news AS hn"
        + join_clause
        + " WHERE hn.latest_seen_time >= '{}' AND hn.latest_seen_time < '{}'{}{} "
        "ORDER BY hn.latest_seen_time DESC LIMIT {}"
    ).format(
        _sql_literal(start),
        _sql_literal(end),
        platform_clause,
        tag_clause,
        safe_limit,
    )


def build_hot_news_rank_sql(
    candidate_ids: Iterable[Any],
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None,
) -> str:
    """生成第二阶段排名 SQL，只访问 hot_news_record，最多查询 20 个候选和 100 行。"""

    ids = _bounded_ints(candidate_ids, MAX_RANKING_IDS)
    if not ids:
        raise ValueError("ranking stage requires at least one positive candidate id")
    start, end = _time_window(start_time, end_time)
    return (
        "SELECT hot_news_id, platform_code, rank_no, record_time "
        "FROM hot_news_record "
        "WHERE hot_news_id IN ({}) "
        "AND record_time >= '{}' AND record_time < '{}' "
        "ORDER BY hot_news_id ASC, record_time DESC LIMIT {}"
    ).format(
        ", ".join(str(item) for item in ids),
        _sql_literal(start),
        _sql_literal(end),
        MAX_RANKING_ROWS,
    )


def _rows_from_proxy(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = result.get("rows") if isinstance(result, dict) else []
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    data = result.get("data") if isinstance(result, dict) else []
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)]
    return []


def _normalize_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "title": row.get("title") or "",
        "url": row.get("url"),
        "platform_code": row.get("platform_code"),
        "platform_name": row.get("platform_name"),
        "first_seen_time": row.get("first_seen_time"),
        "latest_seen_time": row.get("latest_seen_time"),
        "source_type": "database",
    }


def _normalize_ranking(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hot_news_id": row.get("hot_news_id"),
        "platform_code": row.get("platform_code"),
        "rank_no": row.get("rank_no"),
        "record_time": row.get("record_time"),
        "source_type": "database",
    }


def fetch_hot_news(
    stage: str = "candidates",
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None,
    platforms: Optional[Iterable[Any]] = None,
    tag_ids: Optional[Iterable[Any]] = None,
    candidate_ids: Optional[Iterable[Any]] = None,
    limit: int = MAX_CANDIDATES,
    base_url: Optional[str] = None,
    timeout_ms: int = 15_000,
) -> Dict[str, Any]:
    """执行 hot_news_fetcher 的一个阶段，不在一次调用中混合候选与排名查询。"""

    stage_name = str(stage or "candidates").strip().lower()
    start, end = _time_window(start_time, end_time)
    if stage_name == "candidates":
        sql = build_hot_news_sql(start, end, platforms, tag_ids, limit)
        result = query_java_sql_proxy(sql=sql, base_url=base_url, timeout_ms=timeout_ms)
        rows = _rows_from_proxy(result)[:MAX_CANDIDATES]
        return {
            "source_type": "database",
            "source_name": "hot_news_fetcher",
            "stage": "candidates",
            "query_window": {"start_time": start, "end_time": end},
            "records": [_normalize_candidate(row) for row in rows],
            "candidate_ids": [row.get("id") for row in rows if row.get("id") is not None],
            "row_count": len(rows),
            "limit_applied": min(max(1, int(limit or MAX_CANDIDATES)), MAX_CANDIDATES),
        }
    if stage_name == "ranking":
        ids = _bounded_ints(candidate_ids, MAX_RANKING_IDS)
        sql = build_hot_news_rank_sql(ids, start, end)
        result = query_java_sql_proxy(sql=sql, base_url=base_url, timeout_ms=timeout_ms)
        rows = _rows_from_proxy(result)[:MAX_RANKING_ROWS]
        return {
            "source_type": "database",
            "source_name": "hot_news_fetcher",
            "stage": "ranking",
            "query_window": {"start_time": start, "end_time": end},
            "candidate_ids": ids,
            "records": [_normalize_ranking(row) for row in rows],
            "row_count": len(rows),
            "limit_applied": MAX_RANKING_ROWS,
        }
    raise ValueError("hot_news_fetcher stage must be 'candidates' or 'ranking'")
