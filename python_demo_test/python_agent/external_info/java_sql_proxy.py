"""Java 动态 SQL 只读代理工具。

作用：把 Agent 生成的单条只读 SELECT 转发给 Java 后端，并将响应标准化为工具结果。
项目依赖：`view.tooling` 的 BuiltinTool、工具上下文和结果类型。
外部依赖：Python 标准库 urllib；实际数据库访问由 Java 服务完成。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

from ..view.tooling import BuiltinTool, ToolCapabilities, ToolContext, ToolRegistry, ToolResult


JAVA_SQL_PROXY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["sql"],
    "properties": {
        "sql": {
            "type": "string",
            "description": "单条 MySQL SELECT 查询语句，必须包含 LIMIT。",
        },
        "base_url": {
            "type": "string",
            "description": "Java 后端地址，默认读取 JAVA_BACKEND_BASE_URL 或 http://127.0.0.1:8080。",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "请求超时毫秒数，默认 15000。",
        },
    },
    "additionalProperties": True,
}


def query_java_sql_proxy(sql: str, base_url: str | None = None, timeout_ms: int = 15_000) -> Dict[str, Any]:
    sql_text = (sql or "").strip()
    if not sql_text:
        raise ValueError("sql must not be blank")

    root_url = (base_url or os.getenv("JAVA_BACKEND_BASE_URL") or "http://127.0.0.1:8080").rstrip("/")
    url = f"{root_url}/api/sql-proxy/query"
    payload = json.dumps({"sql": sql_text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=max(1, timeout_ms / 1000)) as response:  # noqa: S310 - local backend URL is configured.
            body = response.read().decode("utf-8")
            return json.loads(body or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Java SQL proxy returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to call Java SQL proxy: {exc.reason}") from exc


def execute_java_sql_proxy(
    input: Dict[str, Any],
    _ctx: ToolContext,
    default_base_url: str | None = None,
) -> ToolResult:
    try:
        result = query_java_sql_proxy(
            sql=str(input.get("sql") or ""),
            base_url=input.get("base_url") or default_base_url,
            timeout_ms=int(input.get("timeout_ms") or 15_000),
        )
        return ToolResult(
            content=json.dumps(result, ensure_ascii=False, indent=2),
            metadata={"result": result},
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary converts any failure.
        return ToolResult(
            content=f"java_sql_proxy_query failed: {exc}",
            is_error=True,
            metadata={"error_type": exc.__class__.__name__},
        )


def register_java_sql_proxy_tools(
    registry: ToolRegistry,
    default_base_url: str | None = None,
) -> ToolRegistry:
    registry.register(
        BuiltinTool(
            tool_name="java_sql_proxy_query",
            tool_description=(
                "查询项目 MySQL 中已经采集的热点、标签、平台和排名趋势数据。"
                "只接受一条带 LIMIT 的 SELECT；Java 会执行 SQL 审查、超时、行数限制和限流。"
                "该工具不是公开互联网实时搜索。"
            ),
            schema=JAVA_SQL_PROXY_SCHEMA,
            handler=lambda input, ctx: execute_java_sql_proxy(input, ctx, default_base_url),
            tool_capabilities=ToolCapabilities(read=True, network=True),
            is_read_only=True,
        )
    )
    return registry


def build_java_sql_proxy_tool_registry(default_base_url: str | None = None) -> ToolRegistry:
    return register_java_sql_proxy_tools(ToolRegistry(), default_base_url)
