"""Bocha Web Search 客户端。

作用：调用 Bocha 搜索接口，兼容不同响应结构，并统一转换为搜索记录；同时集中管理直连/代理策略。
项目依赖：被 `web_tools.py` 和 `domain.py` 调用。
外部依赖：Bocha Web Search HTTP API；网络请求使用 Python 标准库 urllib。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


BOCHA_WEB_SEARCH_URL = "https://api.bocha.cn/v1/web-search"


class BochaConfigError(RuntimeError):
    """Bocha API 配置错误。"""


class BochaApiError(RuntimeError):
    """Bocha API 调用错误。"""


class NetworkProxyConfigError(RuntimeError):
    """公共联网工具的代理配置错误。"""


def build_public_network_opener() -> urllib.request.OpenerDirector:
    """创建公共联网工具使用的 opener，默认不继承 Windows 系统代理。

    `urllib.request.urlopen()` 会隐式读取 Windows 系统代理。代理客户端未运行时，
    这会把所有搜索和网页抓取请求导向失效的 127.0.0.1 端口。公共工具因此默认
    `direct` 直连；只有部署方明确配置时才启用代理。

    - `AGENT_NETWORK_PROXY_MODE=direct`（默认）：禁用系统及环境代理。
    - `AGENT_NETWORK_PROXY_MODE=system`：显式继承系统/环境代理。
    - `AGENT_NETWORK_PROXY_MODE=url`：使用 `AGENT_NETWORK_PROXY_URL` 指定的 http(s) 代理。
    """

    mode = os.getenv("AGENT_NETWORK_PROXY_MODE", "direct").strip().lower()
    if mode in ("", "direct", "none"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if mode in ("system", "env"):
        return urllib.request.build_opener(urllib.request.ProxyHandler())
    if mode == "url":
        proxy_url = os.getenv("AGENT_NETWORK_PROXY_URL", "").strip()
        parsed = urllib.parse.urlparse(proxy_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise NetworkProxyConfigError(
                "AGENT_NETWORK_PROXY_URL must be a valid http(s) proxy URL when "
                "AGENT_NETWORK_PROXY_MODE=url"
            )
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
    raise NetworkProxyConfigError(
        "AGENT_NETWORK_PROXY_MODE must be direct, system, or url"
    )


def read_bocha_api_key() -> str:
    """从环境变量或文件读取 Bocha API key。

    优先级：
    1. BOCHA_API_KEY
    2. BOCHA_API_KEY_FILE
    3. python_demo_test/web_api.txt
    4. assert/bocha_api.txt
    """

    direct_key = os.getenv("BOCHA_API_KEY", "").strip()
    if direct_key:
        return direct_key

    candidates = [
        os.getenv("BOCHA_API_KEY_FILE", "").strip(),
        "python_demo_test/web_api.txt",
        "assert/bocha_api.txt",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                return value
    raise BochaConfigError("Bocha API key is missing. Set BOCHA_API_KEY or BOCHA_API_KEY_FILE.")


def bocha_web_search(
    query: str,
    count: int = 10,
    summary: bool = True,
    freshness: Optional[str] = None,
) -> Dict[str, Any]:
    """调用 Bocha Web Search API。"""

    if not query.strip():
        raise ValueError("query is required")

    payload: Dict[str, Any] = {
        "query": query.strip(),
        "summary": bool(summary),
        "count": max(1, min(int(count or 10), 20)),
    }
    if freshness:
        payload["freshness"] = freshness

    disable_invalid_ssl_keylog_file()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BOCHA_WEB_SEARCH_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {read_bocha_api_key()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with build_public_network_opener().open(request, timeout=30) as response:  # noqa: S310 - configured first-party API URL.
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise BochaApiError(f"Bocha API HTTP {exc.code}: {error_text}") from exc
    except urllib.error.URLError as exc:
        raise BochaApiError(f"Bocha API request failed: {exc.reason}") from exc

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise BochaApiError(f"Bocha API returned non-JSON response: {response_text[:500]}") from exc


def disable_invalid_ssl_keylog_file() -> None:
    """移除不可写的 SSLKEYLOGFILE，避免 Windows 本地 SSL 握手失败。"""

    keylog_file = os.getenv("SSLKEYLOGFILE", "").strip()
    if not keylog_file:
        return
    path = Path(keylog_file).expanduser()
    parent = path.parent
    if not parent.exists():
        os.environ.pop("SSLKEYLOGFILE", None)
        return
    try:
        with path.open("a", encoding="utf-8"):
            pass
    except OSError:
        os.environ.pop("SSLKEYLOGFILE", None)


def normalize_bocha_results(raw_response: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """把 Bocha 返回结果整理成 search_records。

    Bocha 返回结构可能随版本变化，这里兼容常见的 data.webPages.value
    和 data.results 两种形态，保留原始 item 供后续排查。
    """

    data = raw_response.get("data") if isinstance(raw_response.get("data"), dict) else {}
    web_pages = data.get("webPages") if isinstance(data.get("webPages"), dict) else {}
    values = web_pages.get("value") or data.get("results") or raw_response.get("results") or []
    records: List[Dict[str, Any]] = []
    for index, item in enumerate(values if isinstance(values, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("name") or item.get("title") or ""
        snippet = item.get("snippet") or item.get("summary") or item.get("description") or ""
        records.append(
            {
                "keyword": query,
                "source_name": "bocha_web_search",
                "rank_no": index,
                "title": title,
                "url": item.get("url"),
                "snippet": snippet,
                "summary": item.get("summary"),
                "site_name": item.get("siteName") or item.get("site_name"),
                "date_published": item.get("datePublished") or item.get("date_published"),
                "raw": item,
            }
        )
    return records


def fetch_web_search_sources(
    queries: List[str],
    count: int = 10,
    summary: bool = True,
    freshness: Optional[str] = None,
) -> Dict[str, Any]:
    """批量联网搜索，并输出 search_records 与 data_sources。"""

    all_records: List[Dict[str, Any]] = []
    raw_responses: List[Dict[str, Any]] = []
    for query in queries:
        raw_response = bocha_web_search(query=query, count=count, summary=summary, freshness=freshness)
        raw_responses.append({"query": query, "response": raw_response})
        all_records.extend(normalize_bocha_results(raw_response, query=query))

    return {
        "search_records": all_records,
        "data_sources": [
            {
                "source_id": "bocha_web_search",
                "source_type": "search",
                "source_name": "Bocha Web Search",
                "records": all_records,
            }
        ],
        "raw_responses": raw_responses,
        "summary": {
            "query_count": len(queries),
            "record_count": len(all_records),
            "provider": "bocha",
        },
    }
