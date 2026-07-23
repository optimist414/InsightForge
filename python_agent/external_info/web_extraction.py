"""网页正文提取、质量审查和压缩模块。

作用：把 HTML 转为结构化正文，并按文本长度调用独立 LLM 做质量判断或压缩。
项目依赖：无本项目模块依赖；由 `web_tools.py` 调用。
外部依赖：`trafilatura` 负责正文抽取，urllib 负责 LLM HTTP 请求。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from trafilatura import bare_extraction
except ImportError:  # pragma: no cover - verified through the public error path.
    bare_extraction = None


RAW_TEXT_THRESHOLD = 5_000
MAX_COMPRESSIBLE_CHARS = 500_000
DEFAULT_QUALITY_PREVIEW_CHARS = 12_000
DEFAULT_SUMMARY_MAX_CHARS = 5_000


class WebExtractionDependencyError(RuntimeError):
    """网页正文提取依赖尚未安装。"""


class WebExtractionLlmError(RuntimeError):
    """审查或压缩模型调用失败。"""


def extract_article(html: str, url: str) -> Dict[str, Any]:
    """用 Trafilatura 从 HTML 中得到面向 Agent 的结构化正文。"""

    if bare_extraction is None:
        raise WebExtractionDependencyError(
            "trafilatura is required for article extraction. "
            "Install python_demo_test/agent_input_framework_mvp/requirements.txt"
        )

    document = bare_extraction(
        html,
        url=url,
        include_comments=False,
        include_links=True,
        include_images=False,
        include_tables=True,
        favor_precision=True,
    )
    if document is None:
        return {
            "success": False,
            "url": url,
            "error_code": "EXTRACTION_FAILED",
            "requires_browser": True,
            "extractor": "trafilatura",
        }

    # Trafilatura 2.0 (Python 3.8 compatible) returns its own Document class;
    # newer versions may expose a dataclass. Support both without changing the
    # structured output contract.
    data = document.as_dict() if hasattr(document, "as_dict") else asdict(document)
    text = (data.get("text") or "").strip()
    return {
        "success": bool(text),
        "url": url,
        "title": data.get("title"),
        "author": data.get("author"),
        "published_at": data.get("date"),
        "description": data.get("description"),
        "site_name": data.get("sitename"),
        "hostname": data.get("hostname"),
        "text": text,
        "categories": data.get("categories"),
        "tags": data.get("tags"),
        "extractor": "trafilatura",
        "requires_browser": not bool(text),
    }


class WebExtractLlmClient:
    """给网页抽取链路专用的 Chat Completions 客户端。

    它与主 Agent 共用 DeepSeek API 凭据，但每个任务都使用独立 system prompt，
    从而把正文可用性判断与正文压缩从主 Agent 的推理循环中隔离开。
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("AGENT_WEB_EXTRACT_BASE_URL", os.getenv("AGENT_LLM_BASE_URL", "https://api.deepseek.com")).strip()
        self.model_name = os.getenv(
            "AGENT_WEB_EXTRACT_MODEL_NAME",
            os.getenv("AGENT_LLM_MODEL_NAME", "deepseek-v4-flash"),
        ).strip()
        self.timeout_seconds = max(10, min(int(os.getenv("AGENT_WEB_EXTRACT_TIMEOUT_SECONDS", "90")), 300))
        self.use_env_proxy = os.getenv("AGENT_WEB_EXTRACT_USE_ENV_PROXY", "false").lower() == "true"
        self.api_key = self._resolve_api_key()

    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model_name)

    def review_quality(self, article: Dict[str, Any]) -> Dict[str, Any]:
        text = str(article.get("text") or "")
        preview_limit = max(1_000, min(int(os.getenv("AGENT_WEB_EXTRACT_QUALITY_PREVIEW_CHARS", str(DEFAULT_QUALITY_PREVIEW_CHARS))), 50_000))
        review_input = {
            "url": article.get("url"),
            "title": article.get("title"),
            "description": article.get("description"),
            "site_name": article.get("site_name"),
            "text_char_count": len(text),
            "text_preview": text[:preview_limit],
        }
        response = self._chat_json(
            system_prompt=(
                "你是网页正文抽取质量审查器。只判断给定结构化抽取结果是否足以支撑后续事实整理。"
                "正文应包含连贯的主题信息，不能主要是导航、登录提示、免责声明、标签或代码。"
                "只返回 JSON 对象，不要 Markdown："
                '{"usable":true,"quality":"high|medium|low","reason":"简短原因","requires_alternate_url":false}'
            ),
            user_payload=review_input,
            max_tokens=300,
        )
        usable = bool(response.get("usable"))
        quality = str(response.get("quality") or ("medium" if usable else "low")).lower()
        if quality not in {"high", "medium", "low"}:
            quality = "medium" if usable else "low"
        return {
            "checked_by": "web_extract_quality_reviewer",
            "model": self.model_name,
            "usable": usable,
            "quality": quality,
            "reason": str(response.get("reason") or "No quality reason returned."),
            "requires_alternate_url": bool(response.get("requires_alternate_url", not usable)),
            "reviewed_text_char_count": min(len(text), preview_limit),
        }

    def compress(self, article: Dict[str, Any]) -> Dict[str, Any]:
        text = str(article.get("text") or "")
        limit = max(500, min(int(os.getenv("AGENT_WEB_EXTRACT_SUMMARY_MAX_CHARS", str(DEFAULT_SUMMARY_MAX_CHARS))), 20_000))
        response = self._chat_json(
            system_prompt=(
                "你是网页正文压缩器。请在不虚构内容的前提下压缩正文，保留事件主体、时间、人物/机构、"
                "数据、因果关系、不同观点、限定条件和不确定性。忽略导航、广告、互动引导和重复段落。"
                f"输出纯文本，不要标题、前言或 Markdown；尽量不超过 {limit} 个中文字符。"
            ),
            user_payload={
                "url": article.get("url"),
                "title": article.get("title"),
                "published_at": article.get("published_at"),
                "description": article.get("description"),
                "article_text": text,
            },
            max_tokens=max(1200, min(8000, limit * 2)),
            expect_json=False,
        )
        summary = str(response).strip()
        if not summary:
            raise WebExtractionLlmError("web_extract compressor returned empty text")
        output_text = summary[:limit]
        return {
            "applied": True,
            "method": "web_extract_llm",
            "model": self.model_name,
            "source_char_count": len(text),
            "output_char_count": len(output_text),
            "max_output_chars": limit,
            "text": output_text,
        }

    def _chat_json(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        max_tokens: int,
        expect_json: bool = True,
    ) -> Any:
        if not self.is_configured():
            raise WebExtractionLlmError("web_extract LLM is not configured")
        payload = {
            "model": self.model_name,
            "stream": False,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            self._chat_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._build_opener().open(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise WebExtractionLlmError(f"web_extract LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise WebExtractionLlmError(f"web_extract LLM request failed: {exc.reason}") from exc
        try:
            body = json.loads(raw_response)
            content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
        except (json.JSONDecodeError, AttributeError, IndexError) as exc:
            raise WebExtractionLlmError("web_extract LLM returned an invalid Chat Completions response") from exc
        if not isinstance(content, str) or not content.strip():
            raise WebExtractionLlmError("web_extract LLM returned no content")
        if not expect_json:
            return content
        return _parse_json_object(content)

    def _chat_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

    def _resolve_api_key(self) -> str:
        direct_key = os.getenv("AGENT_WEB_EXTRACT_API_KEY", os.getenv("AGENT_LLM_API_KEY", "")).strip()
        if direct_key:
            return _normalize_api_key(direct_key)
        key_file = os.getenv("AGENT_WEB_EXTRACT_API_KEY_FILE", os.getenv("AGENT_LLM_API_KEY_FILE", "assert/agent_api.txt")).strip()
        path = Path(key_file)
        if not path.exists():
            return ""
        for line in path.read_text(encoding="utf-8").splitlines():
            key = _normalize_api_key(line)
            if key.startswith("sk-"):
                return key
        return ""

    def _build_opener(self) -> urllib.request.OpenerDirector:
        """沿用主 Agent 的默认策略，避免无效系统代理影响辅助模型。"""

        _disable_invalid_ssl_keylog_file()
        handlers = [] if self.use_env_proxy else [urllib.request.ProxyHandler({})]
        return urllib.request.build_opener(*handlers)


def enrich_article_for_agent(
    article: Dict[str, Any],
    *,
    enable_quality_check: bool,
    enable_compression: bool,
) -> Dict[str, Any]:
    """执行质量审查与按字符长度决定的单次压缩。"""

    result = dict(article)
    raw_text = str(result.get("text") or "")
    result["source_text_char_count"] = len(raw_text)
    result["content"] = raw_text
    result["quality"] = {
        "checked": False,
        "usable": bool(raw_text),
        "quality": "unreviewed",
        "requires_alternate_url": not bool(raw_text),
    }
    result["compression"] = {
        "applied": False,
        "reason": "text_not_available" if not raw_text else "below_threshold",
        "source_char_count": len(raw_text),
        "output_char_count": len(raw_text),
    }
    if not raw_text:
        return result

    llm: Optional[WebExtractLlmClient] = None
    if enable_quality_check:
        try:
            llm = WebExtractLlmClient()
            result["quality"] = {"checked": True, **llm.review_quality(result)}
        except WebExtractionLlmError as exc:
            result["quality"] = {
                "checked": False,
                "usable": True,
                "quality": "unknown",
                "reason": str(exc),
                "review_error": True,
                "requires_alternate_url": False,
            }

    if not result["quality"].get("usable", False):
        result["success"] = False
        result["requires_browser"] = bool(result["quality"].get("requires_alternate_url"))
        result["content"] = ""
        result["compression"] = {
            "applied": False,
            "reason": "quality_rejected",
            "source_char_count": len(raw_text),
            "output_char_count": 0,
        }
        return result

    if not enable_compression or len(raw_text) <= RAW_TEXT_THRESHOLD:
        return result
    if len(raw_text) > MAX_COMPRESSIBLE_CHARS:
        result["compression"] = {
            "applied": False,
            "reason": "source_exceeds_500000_chars",
            "source_char_count": len(raw_text),
            "output_char_count": len(raw_text),
        }
        return result
    try:
        llm = llm or WebExtractLlmClient()
        compressed = llm.compress(result)
        result["content"] = compressed.pop("text")
        result["compression"] = compressed
    except WebExtractionLlmError as exc:
        result["compression"] = {
            "applied": False,
            "reason": "compression_failed",
            "error": str(exc),
            "source_char_count": len(raw_text),
            "output_char_count": len(raw_text),
        }
    return result


def _normalize_api_key(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("sk-"):
        return normalized
    for separator in (":", "="):
        if separator in normalized:
            normalized = normalized.split(separator, 1)[1].strip()
    return normalized


def _disable_invalid_ssl_keylog_file() -> None:
    keylog_file = os.getenv("SSLKEYLOGFILE", "").strip()
    if not keylog_file:
        return
    path = Path(keylog_file).expanduser()
    if not path.parent.exists():
        os.environ.pop("SSLKEYLOGFILE", None)
        return
    try:
        with path.open("a", encoding="utf-8"):
            pass
    except OSError:
        os.environ.pop("SSLKEYLOGFILE", None)


def _parse_json_object(content: str) -> Dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = _extract_first_json_object(cleaned)
    if not isinstance(value, dict):
        raise WebExtractionLlmError("quality reviewer returned a non-object JSON value")
    return value


def _extract_first_json_object(value: str) -> Dict[str, Any]:
    """兼容模型在 JSON 前后额外添加一句说明的情况。"""

    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise WebExtractionLlmError("quality reviewer did not return a JSON object")
