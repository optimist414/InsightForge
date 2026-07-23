"""B 站创作者数据采集适配器。

作用：通过临时、无登录的 Edge CDP 会话读取创作者公开视频列表，供 `creator_profile_builder` 内部使用。
该模块不再作为独立 Agent 工具注册；画像工具负责采集、格式化和调用画像提示词。
项目依赖：`view.tooling` 的失败 DTO、工具上下文和工具结果类型。
外部依赖：`websocket`；运行时需要本机安装 Microsoft Edge。
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import websocket

from ...view.tooling import ToolContext, ToolFailureDTO, ToolResult


BILIBILI_CREATOR_VIDEOS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "anyOf": [
        {"required": ["uid"]},
        {"required": ["creator_url"]},
    ],
    "properties": {
        "uid": {
            "type": "string",
            "description": "B 站创作者 UID（mid），例如 277463278。",
        },
        "creator_url": {
            "type": "string",
            "description": "B 站创作者空间链接，例如 https://space.bilibili.com/277463278。",
        },
        "limit": {
            "type": "integer",
            "description": "最多读取最近发布的视频数，默认 50，最大 50。",
        },
        "include_details": {
            "type": "boolean",
            "default": False,
            "description": "是否进入视频详情页补充简介和标签，默认 false。仅当用户明确要求简介或标签时才可设为 true；画像和选题推荐默认不读取详情。",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "所有浏览器重试合计的读取总超时毫秒数，默认 60000，最大 120000。",
        },
        "max_attempts": {
            "type": "integer",
            "description": "使用全新无登录浏览器会话的最大尝试次数，默认 3，最大 3。",
        },
    },
    "additionalProperties": False,
}


EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)
EDGE_DIRECT_CONNECTION_FLAGS = ("--no-proxy-server",)
SPACE_URL_PATTERN = re.compile(r"(?:https?://)?space\.bilibili\.com/(\d+)", re.IGNORECASE)
BVID_PATTERN = re.compile(r"/video/(BV[0-9A-Za-z]+)", re.IGNORECASE)


def execute_bilibili_creator_videos(input: Dict[str, Any], _ctx: ToolContext) -> ToolResult:
    """用无登录的浏览器页面读取创作者公开视频卡片和详情页元数据。"""

    try:
        uid = _parse_uid(input)
        limit = max(1, min(int(input.get("limit") or 50), 50))
        # 只接受 JSON 布尔真值，避免字符串 "false" 被 Python bool 误判为 true 后打开视频详情页。
        include_details = input.get("include_details") is True
        total_timeout_seconds = max(15, min(int(input.get("timeout_ms") or 60_000), 120_000)) / 1000
        max_attempts = max(1, min(int(input.get("max_attempts") or 3), 3))
        return _collect_with_retry(
            uid=uid,
            limit=limit,
            include_details=include_details,
            total_timeout_seconds=total_timeout_seconds,
            max_attempts=max_attempts,
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary must convert any failure to ToolResult.
        return ToolFailureDTO.from_exception("bilibili_creator_videos", exc).to_result()


def _parse_uid(input: Dict[str, Any]) -> str:
    uid = str(input.get("uid") or "").strip()
    if uid.isdigit():
        return uid

    creator_url = str(input.get("creator_url") or "").strip()
    match = SPACE_URL_PATTERN.search(creator_url)
    if match:
        return match.group(1)
    raise ValueError("bilibili_creator_videos requires a numeric uid or a space.bilibili.com creator_url")


@dataclass
class BilibiliCollectionError(RuntimeError):
    """附带页面状态的可恢复采集失败。"""

    message: str
    diagnostic: Dict[str, Any]

    def __post_init__(self) -> None:
        super().__init__(self.message)


def _collect_with_retry(
    *,
    uid: str,
    limit: int,
    include_details: bool,
    total_timeout_seconds: float,
    max_attempts: int,
) -> ToolResult:
    """使用独立浏览器会话重试，避免一次页面 hydration 失败终止采集。"""

    started_at = time.monotonic()
    diagnostics: List[Dict[str, Any]] = []
    last_error: Optional[Exception] = None

    for attempt_no in range(1, max_attempts + 1):
        remaining_seconds = total_timeout_seconds - (time.monotonic() - started_at)
        attempts_left = max_attempts - attempt_no + 1
        if remaining_seconds < 5:
            break
        attempt_timeout_seconds = max(5, min(30, remaining_seconds / attempts_left))
        crawler = BilibiliCreatorBrowser(
            timeout_seconds=attempt_timeout_seconds,
            attempt_no=attempt_no,
        )
        try:
            payload = crawler.collect(uid=uid, limit=limit, include_details=include_details)
            diagnostics.append(
                payload.pop(
                    "_attempt_diagnostic",
                    {"attempt": attempt_no, "outcome": "success"},
                )
            )
            payload["collection_diagnostics"] = {
                "attempt_count": attempt_no,
                "successful_attempt": attempt_no,
                "attempts": diagnostics,
            }
            return ToolResult(
                content=json.dumps(payload, ensure_ascii=False, indent=2),
                is_error=False,
                metadata={"result": payload, "collection_diagnostics": payload["collection_diagnostics"]},
            )
        except BilibiliCollectionError as exc:
            last_error = exc
            diagnostics.append(exc.diagnostic)
        except Exception as exc:  # noqa: BLE001 - every attempt must leave a diagnosable record.
            last_error = exc
            diagnostics.append(
                {
                    "attempt": attempt_no,
                    "outcome": "runtime_error",
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                }
            )

        if attempt_no < max_attempts and total_timeout_seconds - (time.monotonic() - started_at) > 2:
            time.sleep(min(float(attempt_no), 2.0))

    error = last_error or RuntimeError("Bilibili creator page did not produce a video listing before timeout")
    failure = ToolFailureDTO.from_exception("bilibili_creator_videos", error)
    result = failure.to_result()
    payload = json.loads(result.content)
    payload["diagnostics"] = {
        "attempt_count": len(diagnostics),
        "max_attempts": max_attempts,
        "total_timeout_ms": int(total_timeout_seconds * 1000),
        "attempts": diagnostics,
    }
    result.content = json.dumps(payload, ensure_ascii=False, indent=2)
    result.metadata = {
        **(result.metadata or {}),
        "result": payload,
        "collection_diagnostics": payload["diagnostics"],
    }
    return result


class BilibiliCreatorBrowser:
    """最小 CDP 客户端：不依赖 Selenium，专门读取 B 站动态公开页面。"""

    def __init__(self, timeout_seconds: float, attempt_no: int = 1) -> None:
        self.timeout_seconds = timeout_seconds
        self.attempt_no = attempt_no
        self.process: Optional[subprocess.Popen[Any]] = None
        self.temp_profile: Optional[tempfile.TemporaryDirectory[str]] = None
        self.ws: Optional[websocket.WebSocket] = None
        self.message_id = 0
        self.events: List[Dict[str, Any]] = []
        self.network_failures: List[str] = []
        self.startup_probe_errors: List[str] = []

    def collect(self, uid: str, limit: int, include_details: bool) -> Dict[str, Any]:
        try:
            self._start()
            space_url = f"https://space.bilibili.com/{uid}/upload/video"
            self._navigate(space_url)
            cards, source = self._wait_for_video_listing(limit, include_details)
            if not cards:
                raise BilibiliCollectionError(
                    "No video cards appeared on the creator space page",
                    self.diagnostics(outcome="no_video_listing"),
                )
            used_visible_text_fallback = source == "visible_text" or any(not card.get("url") for card in cards)
            videos: List[Dict[str, Any]] = []
            details_failures: List[Dict[str, str]] = []
            for card in cards:
                video = {
                    "bvid": card["bvid"],
                    "title": card.get("title") or "",
                    "url": card["url"],
                    "description": "",
                    "tags": [],
                    "source": card.get("source") or "bilibili_browser_page",
                }
                for field in ("published_at", "duration", "view_count_text", "engagement_count_text"):
                    if card.get(field):
                        video[field] = card[field]
                if include_details and card.get("url"):
                    try:
                        video.update(self._read_video_detail(card["url"]))
                    except Exception as exc:  # noqa: BLE001 - one failed video must not discard the creator result.
                        details_failures.append({"bvid": card["bvid"], "error": str(exc)})
                videos.append(video)

            payload = {
                "success": bool(videos),
                "platform": "bilibili",
                "uid": uid,
                "creator_url": f"https://space.bilibili.com/{uid}",
                "video_count": len(videos),
                "videos": videos,
                "partial_failures": details_failures,
                "retrieved_at": int(time.time()),
                "notes": [
                    "通过未登录的 Edge 无头浏览器读取公开页面，不使用 Cookie 或账号凭据。",
                    "空间页只加载当前可见视频卡片；limit 较大时结果仍可能受页面分页、动态加载或访问限制影响。",
                ]
                + (
                    [
                        "此账号的新版空间卡片未暴露可读取链接，已从可见文本降级提取标题、日期、时长和公开统计。"
                        "该模式无法获取 bvid、简介和标签。"
                    ]
                    if used_visible_text_fallback
                    else []
                )
                + [f"本次读取策略：{source}。", f"本次浏览器会话尝试序号：{self.attempt_no}。"],
            }
            payload["_attempt_diagnostic"] = self.diagnostics(outcome="success")
            return payload
        finally:
            self.close()

    def _wait_for_video_listing(self, limit: int, include_details: bool) -> Tuple[List[Dict[str, str]], str]:
        """在一个固定窗口内轮询页面、接口和可见文本，避免串行多次长等待。"""

        deadline = time.monotonic() + max(2, min(self.timeout_seconds - 3, 25))
        while time.monotonic() < deadline:
            if include_details:
                cards = self._read_space_cards(limit)
                if cards:
                    return cards, "page_card_links"
                cards = self._read_network_video_listing(limit)
                if cards:
                    return cards, "network_response"
                cards = self._read_visible_video_listing(limit)
                if cards:
                    return cards, "visible_text"
            else:
                # 新版空间卡片的 anchor 文本有时只含统计值；默认画像优先使用稳定的可见标题文本。
                cards = self._read_visible_video_listing(limit)
                if cards:
                    return cards, "visible_text"
                cards = self._read_space_cards(limit)
                if cards:
                    return cards, "page_card_links"
                cards = self._read_network_video_listing(limit)
                if cards:
                    return cards, "network_response"
            time.sleep(0.75)
        return [], "none"

    def diagnostics(self, outcome: str) -> Dict[str, Any]:
        """记录无敏感内容的页面状态，帮助上层判断重试或换源。"""

        self._drain_events()
        page_title = self._diagnostic_evaluate("document.title")
        body_text = self._diagnostic_evaluate("document.body.innerText")
        body_text = body_text if isinstance(body_text, str) else ""
        video_anchor_count = self._diagnostic_evaluate(
            "document.querySelectorAll('a[href*=\"/video/BV\"]').length"
        )
        all_anchor_count = self._diagnostic_evaluate("document.querySelectorAll('a').length")
        arc_search_statuses: List[int] = []
        security_statuses: List[int] = []
        for event in self.events:
            if event.get("method") != "Network.responseReceived":
                continue
            response = (event.get("params") or {}).get("response") or {}
            url = str(response.get("url") or "")
            status = response.get("status")
            if "x/space/wbi/arc/search" in url and isinstance(status, (int, float)):
                arc_search_statuses.append(int(status))
            if "gaia" in url and isinstance(status, (int, float)):
                security_statuses.append(int(status))
        return {
            "attempt": self.attempt_no,
            "outcome": outcome,
            "page_title": str(page_title or "")[:200],
            "body_char_count": len(body_text),
            "has_latest_release": "最新发布" in body_text,
            "has_video_tab": "TA的视频" in body_text,
            "has_login_prompt": "立即登录" in body_text,
            "video_anchor_count": video_anchor_count if isinstance(video_anchor_count, int) else 0,
            "all_anchor_count": all_anchor_count if isinstance(all_anchor_count, int) else 0,
            "arc_search_statuses": arc_search_statuses,
            "security_statuses": security_statuses,
            "security_rejections": list(dict.fromkeys(self.network_failures)),
        }

    def _diagnostic_evaluate(self, expression: str) -> Any:
        try:
            return self._evaluate(expression)
        except Exception:  # noqa: BLE001 - diagnostics must not mask the primary tool failure.
            return None

    def _start(self) -> None:
        edge_path = next((path for path in EDGE_CANDIDATES if path.exists()), None)
        if edge_path is None:
            raise RuntimeError("Microsoft Edge was not found; install Edge or configure a browser adapter")
        cache_root = Path.cwd() / ".cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        self.temp_profile = tempfile.TemporaryDirectory(
            prefix="bilibili_creator_cdp_",
            dir=str(cache_root),
        )
        port = _reserve_local_port()
        command = [
            str(edge_path),
            "--headless=new",
            # 当前 Windows 受限运行环境会阻止 Edge 的默认 sandbox 建立子进程。
            # 此工具只运行本机临时、未登录的展示浏览器，因此 demo 使用 no-sandbox。
            "--no-sandbox",
            "--remote-debugging-address=127.0.0.1",
            # CDP 仅绑定 127.0.0.1；允许本工具生成的本地 WebSocket Origin 完成握手。
            "--remote-allow-origins=*",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={self.temp_profile.name}",
            "--no-first-run",
            "--no-default-browser-check",
            # Edge 默认会继承 Windows 系统代理；采集工具必须保持与 Python 联网工具一致的直连策略。
            *EDGE_DIRECT_CONNECTION_FLAGS,
            "--disable-gpu",
            "--mute-audio",
            "--autoplay-policy=user-gesture-required",
            "--window-size=1440,1200",
            "about:blank",
        ]
        self.process = subprocess.Popen(  # noqa: S603 - executable path is selected from fixed local candidates.
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        endpoint = f"http://127.0.0.1:{port}/json/list"
        deadline = time.monotonic() + min(self.timeout_seconds, 15)
        while time.monotonic() < deadline:
            try:
                # 当前机器存在失效的全局代理；CDP 是本机回环地址，绝不能经过代理。
                with _local_no_proxy_opener().open(endpoint, timeout=1) as response:
                    targets = json.loads(response.read().decode("utf-8"))
                page = next((item for item in targets if item.get("type") == "page"), None)
                if page and page.get("webSocketDebuggerUrl"):
                    self.ws = websocket.create_connection(
                        page["webSocketDebuggerUrl"],
                        timeout=self.timeout_seconds,
                        http_no_proxy=["127.0.0.1", "localhost"],
                    )
                    self._command("Network.enable", {})
                    return
            except Exception as exc:  # noqa: BLE001 - Edge startup is retried until deadline.
                self.startup_probe_errors.append(f"{exc.__class__.__name__}: {exc}")
                self.startup_probe_errors = self.startup_probe_errors[-3:]
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Edge exited before its CDP endpoint became available "
                        f"(exit_code={self.process.returncode}; "
                        f"last_probe_error={self.startup_probe_errors[-1]})"
                    ) from exc
                time.sleep(0.25)
        last_probe_error = self.startup_probe_errors[-1] if self.startup_probe_errors else "none"
        raise RuntimeError(
            "Edge CDP endpoint did not start in time "
            f"(last_probe_error={last_probe_error})"
        )

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup only.
                pass
            self.ws = None
        if self.process is not None:
            self._stop_browser_process_tree(self.process)
            self.process = None
        if self.temp_profile is not None:
            for _ in range(5):
                try:
                    self.temp_profile.cleanup()
                    break
                except PermissionError:
                    time.sleep(0.5)
            self.temp_profile = None

    @staticmethod
    def _stop_browser_process_tree(process: subprocess.Popen[Any]) -> None:
        """关闭当前临时 Edge 及其子进程，避免中断后遗留 crashpad/utility 进程。"""

        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def _navigate(self, url: str) -> None:
        self._command("Page.navigate", {"url": url})
        time.sleep(3)

    def _wait_for_space_cards(self) -> None:
        deadline = time.monotonic() + min(self.timeout_seconds, 20)
        while time.monotonic() < deadline:
            card_count = self._evaluate(
                "document.querySelectorAll('a[href*=\"/video/BV\"]').length"
            )
            if isinstance(card_count, int) and card_count > 0:
                return
            time.sleep(1)
        raise RuntimeError("No video cards appeared on the creator space page")

    def _read_space_cards(self, limit: int) -> List[Dict[str, str]]:
        raw = self._evaluate(
            """
            (() => {
              const result = [];
              const seen = new Set();
              for (const anchor of document.querySelectorAll('a[href*="/video/BV"]')) {
                const match = anchor.href.match(/\\/video\\/(BV[0-9A-Za-z]+)/i);
                if (!match || seen.has(match[1])) continue;
                seen.add(match[1]);
                const title = (anchor.getAttribute('title') ||
                  anchor.querySelector('[title]')?.getAttribute('title') ||
                  anchor.textContent || '').replace(/\\s+/g, ' ').trim();
                result.push({ bvid: match[1], title, url: anchor.href });
              }
              return JSON.stringify(result);
            })()
            """
        )
        records = _parse_json_text(raw, "space video cards")
        return [
            {
                "bvid": str(item.get("bvid") or ""),
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
            }
            for item in records
            if isinstance(item, dict) and item.get("bvid") and item.get("url")
        ][:limit]

    def _read_visible_video_listing(self, limit: int) -> List[Dict[str, str]]:
        """从新版闭合组件的可见文本中提取首屏视频列表。

        部分空间页将卡片放在不可遍历的组件树中，无法获取链接和 bvid；
        此降级路径只保留页面可见的标题、日期、时长和两项统计，避免将它们误认为详情数据。
        """

        raw_text = self._evaluate("document.body.innerText")
        if not isinstance(raw_text, str):
            return []
        return _parse_visible_video_listing(raw_text, limit)

    def _wait_for_visible_video_listing(self, limit: int) -> List[Dict[str, str]]:
        """等待空间页可见列表稳定渲染；不依赖卡片链接或详情页。"""

        deadline = time.monotonic() + min(self.timeout_seconds, 30)
        while time.monotonic() < deadline:
            records = self._read_visible_video_listing(limit)
            if records:
                return records
            time.sleep(0.75)
        return []

    def _read_network_video_listing(self, limit: int) -> List[Dict[str, str]]:
        """从空间页自身已加载的 XHR/Fetch 响应中提取含 bvid 的视频列表。"""

        self._drain_events()
        request_ids = []
        for event in self.events:
            if event.get("method") != "Network.responseReceived":
                continue
            params = event.get("params") or {}
            response = params.get("response") or {}
            url = str(response.get("url") or "")
            if "bilibili.com" not in url or not any(marker in url for marker in ("space", "arc", "video")):
                continue
            request_id = str(params.get("requestId") or "")
            if request_id and request_id not in request_ids:
                request_ids.append(request_id)

        records: List[Dict[str, str]] = []
        seen_bvids = set()
        for request_id in request_ids:
            try:
                response = self._command("Network.getResponseBody", {"requestId": request_id})
                body = str(response.get("body") or "")
                if _is_bilibili_security_rejection(body):
                    self.network_failures.append("space video-list response was rejected with Bilibili error 412")
                    continue
                payload = json.loads(body)
            except Exception:  # noqa: BLE001 - some browser responses cannot expose a body after navigation.
                continue
            for record in _extract_bilibili_video_records(payload):
                bvid = record["bvid"]
                if bvid in seen_bvids:
                    continue
                seen_bvids.add(bvid)
                records.append(record)
                if len(records) >= limit:
                    return records
        return records

    def _read_video_detail(self, url: str) -> Dict[str, Any]:
        self._navigate(url)
        self._wait_for_video_title()
        raw = self._evaluate(
            """
            (() => {
              const textOf = (selectors) => {
                for (const selector of selectors) {
                  const node = document.querySelector(selector);
                  const text = node?.textContent?.replace(/\\s+/g, ' ').trim();
                  if (text) return text;
                }
                return '';
              };
              const state = window.__INITIAL_STATE__ || window.__INITIAL_STATE__ || {};
              const videoData = state.videoData || state.videoInfo || {};
              const desc = videoData.desc || textOf(['#v_desc', '.video-desc-container', '.desc-info-text']);
              const title = videoData.title || textOf(['h1.video-title', 'h1']) || document.title.replace(/_哔哩哔哩_bilibili$/, '').trim();
              const candidates = [
                ...(Array.isArray(state.tags) ? state.tags : []),
                ...(Array.isArray(videoData.tags) ? videoData.tags : []),
              ];
              const tags = candidates.map((item) => item.tag_name || item.name || item).filter(Boolean);
              for (const node of document.querySelectorAll('a[href*="/tag/"], a[href*="topic_detail"]')) {
                const tag = node.textContent?.replace(/\\s+/g, ' ').trim();
                if (tag) tags.push(tag);
              }
              return JSON.stringify({
                title,
                description: String(desc || '').trim(),
                tags: [...new Set(tags.map(String))].slice(0, 30),
              });
            })()
            """
        )
        detail = _parse_json_text(raw, "video detail")
        if not isinstance(detail, dict):
            raise RuntimeError("Unexpected video detail payload")
        return {
            "title": str(detail.get("title") or ""),
            "description": str(detail.get("description") or ""),
            "tags": [str(tag) for tag in detail.get("tags") or [] if str(tag).strip()],
        }

    def _wait_for_video_title(self) -> None:
        deadline = time.monotonic() + min(self.timeout_seconds, 15)
        while time.monotonic() < deadline:
            title = self._evaluate(
                "(document.querySelector('h1.video-title, h1')?.textContent || '').trim()"
            )
            if isinstance(title, str) and title:
                return
            time.sleep(0.75)
        raise RuntimeError("Video detail page did not render a title")

    def _evaluate(self, expression: str) -> Any:
        response = self._command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        # CDP Runtime.evaluate 的 command result 已经是 {result: {type, value}}；
        # _command 返回的是外层 result，因此这里不能再多取一次 result。
        result = response.get("result") or {}
        if "value" in result:
            return result["value"]
        return None

    def _command(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("Browser WebSocket is not connected")
        self.message_id += 1
        request_id = self.message_id
        self.ws.send(json.dumps({"id": request_id, "method": method, "params": params}))
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            message = json.loads(self.ws.recv())
            if message.get("id") != request_id:
                self.events.append(message)
                continue
            if message.get("error"):
                raise RuntimeError(f"CDP {method} failed: {message['error']}")
            return message.get("result") or {}
        raise TimeoutError(f"CDP {method} timed out")

    def _drain_events(self) -> None:
        """取出已经到达的 CDP 事件，避免网络响应只在后续命令时才被读取。"""

        if self.ws is None:
            return
        self.ws.settimeout(0.05)
        try:
            while True:
                self.events.append(json.loads(self.ws.recv()))
        except websocket.WebSocketTimeoutException:
            pass
        finally:
            self.ws.settimeout(self.timeout_seconds)


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _local_no_proxy_opener() -> urllib.request.OpenerDirector:
    """为 CDP 本地端点创建不继承系统 HTTP(S)_PROXY 的 opener。"""

    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _parse_json_text(raw: Any, label: str) -> Any:
    if not isinstance(raw, str):
        raise RuntimeError(f"{label} was not returned as JSON text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not decode {label}: {exc}") from exc


VISIBLE_DATE_PATTERN = re.compile(r"^(?:\d{2}-\d{2}|\d{4}-\d{2}-\d{2})$")
DURATION_PATTERN = re.compile(r"^\d{1,2}:\d{2}$")


def _parse_visible_video_listing(raw_text: str, limit: int) -> List[Dict[str, str]]:
    """解析 B 站空间页可见文本中紧邻发布日期的卡片字段。"""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    records: List[Dict[str, str]] = []
    for index, published_at in enumerate(lines):
        if not VISIBLE_DATE_PATTERN.fullmatch(published_at) or index < 4:
            continue
        duration = lines[index - 2]
        title = lines[index - 1]
        if not DURATION_PATTERN.fullmatch(duration) or not title:
            continue
        records.append(
            {
                "bvid": "",
                "title": title,
                "url": "",
                "published_at": published_at,
                "duration": duration,
                "view_count_text": lines[index - 4],
                "engagement_count_text": lines[index - 3],
            }
        )
        if len(records) >= limit:
            break
    return records


def _extract_bilibili_video_records(payload: Any) -> List[Dict[str, str]]:
    """递归扫描 B 站列表响应，兼容 data.list.vlist 等不同响应嵌套。"""

    records: List[Dict[str, str]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            bvid = str(value.get("bvid") or "").strip()
            title = _strip_html(str(value.get("title") or "")).strip()
            if bvid and title:
                records.append(
                    {
                        "bvid": bvid,
                        "title": title,
                        "url": f"https://www.bilibili.com/video/{bvid}",
                        "source": "bilibili_browser_network_response",
                        "published_at": str(value.get("created") or value.get("pubdate") or ""),
                        "duration": str(value.get("length") or value.get("duration") or ""),
                        "view_count_text": str(value.get("play") or value.get("stat", {}).get("view") or ""),
                        "engagement_count_text": str(value.get("comment") or ""),
                    }
                )
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return records


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).replace("&amp;", "&").strip()


def _is_bilibili_security_rejection(body: str) -> bool:
    return "错误号: 412" in body or "bilibili security control policy" in body

