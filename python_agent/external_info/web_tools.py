"""通用联网工具。

作用：提供 Bocha 网页搜索和已知 URL 抓取，并把网页内容交给正文提取链路处理。
项目依赖：`self_media_topic.bocha_client` 提供搜索与网络策略，`web_extraction` 提供 HTML 提取。
外部依赖：Bocha API、公开 HTTP/HTTPS 网络和可选的 Trafilatura/LLM 服务。
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from ..view.tooling import BuiltinTool, ToolCapabilities, ToolContext, ToolRegistry, ToolResult
from .self_media_topic.bocha_client import (
    bocha_web_search,
    build_public_network_opener,
    disable_invalid_ssl_keylog_file,
    normalize_bocha_results,
)
from .web_extraction import extract_article, enrich_article_for_agent


WEB_SEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "anyOf": [
        {"required": ["query"]},
        {"required": ["q"]},
        {"required": ["search_query"]},
    ],
    "properties": {
        "query": {"type": "string", "description": "搜索词。"},
        "q": {"type": "string", "description": "query 的兼容别名。"},
        "search_query": {
            "type": "array",
            "description": "兼容 web.run 风格的批量搜索输入。",
            "items": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "count": {"type": "integer"},
                },
                "additionalProperties": True,
            },
        },
        "max_results": {"type": "integer", "description": "最多返回结果数，默认 5，最大 20。"},
        "count": {"type": "integer", "description": "max_results 的兼容别名。"},
        "summary": {"type": "boolean", "description": "是否请求 Bocha 摘要，默认 true。"},
        "freshness": {"type": "string", "description": "可选时效过滤参数，透传给 Bocha。"},
    },
    "additionalProperties": True,
}

FETCH_URL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["url"],
    "properties": {
        "url": {"type": "string", "description": "要读取的 http/https URL。"},
        "max_bytes": {"type": "integer", "description": "最多读取字节数，默认 1000000。"},
        "timeout_ms": {"type": "integer", "description": "请求超时毫秒数，默认 15000。"},
        "quality_check": {
            "type": "boolean",
            "description": "HTML 正文是否交给独立审查模型判断可用性，默认 true。",
        },
        "compress": {
            "type": "boolean",
            "description": "正文超过 5000 字符时是否交给 web_extract 模型压缩，默认 true。",
        },
    },
    "additionalProperties": True,
}


def _extract_search_requests(input: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(input.get("search_query"), list):
        requests = []
        for item in input["search_query"]:
            if not isinstance(item, dict):
                continue
            query = str(item.get("q") or item.get("query") or "").strip()
            if not query:
                continue
            requests.append(
                {
                    "query": query,
                    "max_results": int(item.get("max_results") or item.get("count") or input.get("max_results") or input.get("count") or 5),
                }
            )
        return requests

    query = str(input.get("query") or input.get("q") or "").strip()
    if not query:
        raise ValueError("web_search requires query, q, or search_query[].q")
    return [
        {
            "query": query,
            "max_results": int(input.get("max_results") or input.get("count") or 5),
        }
    ]


def execute_web_search(input: Dict[str, Any], _ctx: ToolContext) -> ToolResult:
    """执行 Bocha 搜索，并返回 provider 无关的 SearchResult 结构。"""

    try:
        summary = bool(input.get("summary", True))
        freshness = input.get("freshness")
        search_requests = _extract_search_requests(input)
        all_results = []
        raw_responses = []

        for request in search_requests:
            query = request["query"]
            max_results = max(1, min(int(request["max_results"]), 20))
            raw_response = bocha_web_search(
                query=query,
                count=max_results,
                summary=summary,
                freshness=freshness,
            )
            raw_responses.append({"query": query, "response": raw_response})
            normalized_records = normalize_bocha_results(raw_response, query=query)
            for record in normalized_records[:max_results]:
                all_results.append(
                    {
                        "query": query,
                        "title": record.get("title") or "",
                        "url": record.get("url"),
                        "snippet": record.get("snippet") or record.get("summary") or "",
                        "source_name": record.get("site_name") or record.get("source_name"),
                        "rank_no": record.get("rank_no"),
                    }
                )

        payload = {
            "query": search_requests[0]["query"] if len(search_requests) == 1 else None,
            "queries": [request["query"] for request in search_requests],
            "source": "bocha",
            "count": len(all_results),
            "message": f"Found {len(all_results)} results",
            "results": all_results,
        }
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            metadata={"result": payload, "raw_responses": raw_responses},
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary must convert any failure to ToolResult.
        return ToolResult(
            content=f"web_search failed: {exc}",
            is_error=True,
            metadata={"error_type": exc.__class__.__name__, "provider": "bocha"},
        )


def execute_fetch_url(input: Dict[str, Any], _ctx: ToolContext) -> ToolResult:
    """读取已知 URL，并输出经过质量审查的结构化正文。"""

    try:
        url = str(input.get("url") or "").strip()
        if not url:
            raise ValueError("fetch_url requires url")
        _validate_public_http_url(url)

        max_bytes = max(1, min(int(input.get("max_bytes") or 1_000_000), 5_000_000))
        timeout_seconds = max(1, min(int(input.get("timeout_ms") or 15_000), 60_000)) / 1000
        disable_invalid_ssl_keylog_file()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html,application/json,text/plain,*/*",
            },
        )
        with build_public_network_opener().open(request, timeout=timeout_seconds) as response:  # noqa: S310 - URL is validated.
            status = getattr(response, "status", 200)
            headers = dict(response.headers.items())
            content_type = response.headers.get("Content-Type", "")
            body = response.read(max_bytes + 1)

        truncated = len(body) > max_bytes
        body = body[:max_bytes]
        text = body.decode(_charset_from_content_type(content_type), errors="replace")
        payload: Dict[str, Any] = {
            "url": url,
            "status": status,
            "content_type": content_type,
            "headers": headers,
            "truncated": truncated,
        }
        if "html" not in content_type.lower():
            payload["success"] = bool(text.strip())
            payload["requires_browser"] = False
            payload["content"] = text
            payload["extraction"] = {
                "success": bool(text.strip()),
                "extractor": "plain_text",
                "quality": {"checked": False, "usable": bool(text.strip())},
                "compression": {"applied": False, "reason": "non_html_response"},
            }
        else:
            article = extract_article(text, url)
            enriched = enrich_article_for_agent(
                article,
                enable_quality_check=bool(input.get("quality_check", True)),
                enable_compression=bool(input.get("compress", True)),
            )
            payload["success"] = bool(enriched.get("success"))
            payload["requires_browser"] = bool(enriched.get("requires_browser"))
            payload["content"] = str(enriched.get("content") or "")
            payload["extraction"] = _article_metadata_for_output(enriched)

        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            metadata={"result": payload},
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary must convert any failure to ToolResult.
        return ToolResult(
            content=f"fetch_url failed: {exc}",
            is_error=True,
            metadata={"error_type": exc.__class__.__name__},
        )


def _charset_from_content_type(content_type: str) -> str:
    match = re.search(r"charset=([^;\s]+)", content_type or "", flags=re.IGNORECASE)
    return match.group(1) if match else "utf-8"


def _article_metadata_for_output(article: Dict[str, Any]) -> Dict[str, Any]:
    """避免把压缩前正文重复写入 ToolResult，同时保留可追溯结构化字段。"""

    fields = (
        "success",
        "url",
        "title",
        "author",
        "published_at",
        "description",
        "site_name",
        "hostname",
        "categories",
        "tags",
        "extractor",
        "requires_browser",
        "error_code",
        "source_text_char_count",
        "quality",
        "compression",
    )
    return {field: article.get(field) for field in fields if field in article}


def _validate_public_http_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL hostname is required")
    hostname = parsed.hostname.lower()
    if hostname in ("localhost",) or hostname.endswith(".local"):
        raise ValueError("Localhost URLs are not allowed")
    for address_info in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80)):
        ip = ipaddress.ip_address(address_info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise ValueError(f"Non-public address is not allowed: {ip}")


def register_web_tools(registry: ToolRegistry) -> ToolRegistry:
    """注册通用联网工具。搜索 provider 固定为 Bocha。"""

    registry.register(
        BuiltinTool(
            tool_name="web_search",
            tool_description=(
                "使用 Bocha 搜索公开互联网，返回标题、URL、摘要和来源。"
                "用户要求联网、最新、今天或数据库数据过旧/不足时使用。"
            ),
            schema=WEB_SEARCH_SCHEMA,
            handler=execute_web_search,
            tool_capabilities=ToolCapabilities(read=True, network=True),
            is_read_only=True,
        )
    )
    registry.register(
        BuiltinTool(
            tool_name="fetch_url",
            tool_description=(
                "读取一个已知的公开 http/https URL，并使用 Trafilatura 提取结构化正文，"
                "由独立审查模型判断内容可用性，并在正文超过 5000 字符时压缩。"
                "通常在 web_search 返回 URL 后按需调用；不提供原始 HTML、JSON 或旧版去标签文本模式。"
                "requires_browser=true 时请换链接或使用浏览器能力。"
            ),
            schema=FETCH_URL_SCHEMA,
            handler=execute_fetch_url,
            tool_capabilities=ToolCapabilities(read=True, network=True),
            is_read_only=True,
        )
    )
    return registry


def build_web_tool_registry() -> ToolRegistry:
    return register_web_tools(ToolRegistry())
