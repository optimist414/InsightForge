from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from .core import LlmClient, StreamEvent, StreamEventType, Usage


@dataclass
class ChatCompletionsSseConfig:
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    api_key_file: str = "assert/agent_api.txt"
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.1
    # 0 表示不向服务端发送 max_tokens，由模型/API 使用自身上限。
    max_tokens: int = 0
    timeout_seconds: int = 90
    use_env_proxy: bool = False

    @classmethod
    def from_env(cls) -> "ChatCompletionsSseConfig":
        return cls(
            base_url=os.getenv("AGENT_LLM_BASE_URL", "https://api.deepseek.com").strip(),
            api_key=os.getenv("AGENT_LLM_API_KEY", "").strip(),
            api_key_file=os.getenv("AGENT_LLM_API_KEY_FILE", "assert/agent_api.txt").strip(),
        model_name=os.getenv("AGENT_LLM_MODEL_NAME", "deepseek-v4-flash").strip(),
            temperature=float(os.getenv("AGENT_LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("AGENT_LLM_MAX_TOKENS", "0") or 0),
            timeout_seconds=int(os.getenv("AGENT_LLM_TIMEOUT_SECONDS", "90")),
            use_env_proxy=os.getenv("AGENT_LLM_USE_ENV_PROXY", "false").lower() == "true",
        )

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self._normalize_api_key(self.api_key)
        path = Path(self.api_key_file)
        if not path.exists():
            return ""
        for line in path.read_text(encoding="utf-8").splitlines():
            normalized = self._normalize_api_key(line)
            if normalized.startswith("sk-"):
                return normalized
        return ""

    def chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _normalize_api_key(self, value: str) -> str:
        normalized = (value or "").strip()
        if normalized.startswith("sk-"):
            return normalized
        for separator in (":", "="):
            if separator in normalized:
                normalized = normalized.split(separator, 1)[1].strip()
        return normalized


class ChatCompletionsSseClient(LlmClient):
    """OpenAI-compatible Chat Completions SSE client.

    It converts provider SSE chunks into engine `StreamEvent`s:
    - `delta.content` -> text content block deltas
    - `delta.tool_calls[].function.arguments` -> tool-use input deltas
    """

    def __init__(self, config: Optional[ChatCompletionsSseConfig] = None) -> None:
        self.config = config or ChatCompletionsSseConfig.from_env()

    def create_message_stream(self, request: Dict[str, Any]) -> Iterable[StreamEvent]:
        api_key = self.config.resolve_api_key()
        if not api_key:
            yield StreamEvent(StreamEventType.ERROR, error="Chat Completions API key is not configured")
            return

        body = dict(request)
        body["model"] = self.config.model_name or body.get("model")
        body["stream"] = True
        if body.get("tools"):
            body.setdefault("parallel_tool_calls", False)
        body.setdefault("temperature", self.config.temperature)
        if self.config.max_tokens > 0:
            body.setdefault("max_tokens", self.config.max_tokens)

        http_request = urllib.request.Request(
            self.config.chat_completions_url(),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )

        yield StreamEvent(StreamEventType.MESSAGE_START)
        try:
            opener = self._build_opener()
            with opener.open(http_request, timeout=self.config.timeout_seconds) as response:
                yield from self._stream_response_to_events(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            yield StreamEvent(StreamEventType.ERROR, error=f"Chat Completions HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            yield StreamEvent(StreamEventType.ERROR, error=f"Chat Completions request failed: {exc.reason}")

    def _stream_response_to_events(self, response: Any) -> Iterator[StreamEvent]:
        text_started = False
        tool_started: Dict[int, bool] = {}
        tool_ids: Dict[int, str] = {}
        tool_names: Dict[int, str] = {}
        usage: Optional[Usage] = None

        for payload in self._iter_sse_payloads(response):
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError as exc:
                yield StreamEvent(StreamEventType.ERROR, error=f"Invalid SSE JSON chunk: {exc.msg}")
                continue

            if chunk.get("usage"):
                usage = self._parse_usage(chunk["usage"])

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning_content = delta.get("reasoning_content")
                if reasoning_content:
                    yield StreamEvent(StreamEventType.THINKING_DELTA, delta=str(reasoning_content))
                content = delta.get("content")
                if content:
                    if not text_started:
                        text_started = True
                        yield StreamEvent(StreamEventType.CONTENT_BLOCK_START, block_type="text")
                    yield StreamEvent(StreamEventType.CONTENT_BLOCK_DELTA, block_type="text", delta=content)

                for tool_call in delta.get("tool_calls") or []:
                    index = int(tool_call.get("index", 0))
                    function = tool_call.get("function") or {}
                    if tool_call.get("id"):
                        tool_ids[index] = str(tool_call["id"])
                    if function.get("name"):
                        tool_names[index] = str(function["name"])
                    if not tool_started.get(index) and tool_names.get(index):
                        tool_started[index] = True
                        yield StreamEvent(
                            StreamEventType.CONTENT_BLOCK_START,
                            block_type="tool_use",
                            tool_use_id=tool_ids.get(index, f"call_{index}"),
                            tool_name=tool_names.get(index, ""),
                            tool_call_index=index,
                        )
                    if function.get("arguments"):
                        yield StreamEvent(
                            StreamEventType.CONTENT_BLOCK_DELTA,
                            block_type="tool_use",
                            tool_call_index=index,
                            delta=str(function["arguments"]),
                        )

                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    if text_started:
                        yield StreamEvent(StreamEventType.CONTENT_BLOCK_STOP, block_type="text")
                        text_started = False
                    for index in list(tool_started):
                        if tool_started[index]:
                            yield StreamEvent(
                                StreamEventType.CONTENT_BLOCK_STOP,
                                block_type="tool_use",
                                tool_call_index=index,
                            )
                            tool_started[index] = False

        if text_started:
            yield StreamEvent(StreamEventType.CONTENT_BLOCK_STOP, block_type="text")
        for index, started in tool_started.items():
            if started:
                yield StreamEvent(
                    StreamEventType.CONTENT_BLOCK_STOP,
                    block_type="tool_use",
                    tool_call_index=index,
                )
        yield StreamEvent(StreamEventType.MESSAGE_STOP, usage=usage)

    def _iter_sse_payloads(self, response: Any) -> Iterator[str]:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            yield line[len("data:") :].strip()

    def _parse_usage(self, payload: Dict[str, Any]) -> Usage:
        return Usage(
            input_tokens=int(payload.get("prompt_tokens", 0) or 0),
            output_tokens=int(payload.get("completion_tokens", 0) or 0),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
        )

    def _build_opener(self) -> urllib.request.OpenerDirector:
        self._disable_invalid_ssl_keylog_file()
        handlers = []
        if not self.config.use_env_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener(*handlers)

    def _disable_invalid_ssl_keylog_file(self) -> None:
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
