# 联网正文提取、审查与压缩设计

## 1. 目标与边界

旧版 `fetch_url` 的“原始响应 / JSON / 正则去标签文本”分支已删除。当前 `fetch_url` 唯一职责是下载页面并给 Agent 提供可信、受控大小的正文信息。

链路只处理公开 `http/https` URL，仍沿用原有 DNS 解析与内网地址拦截。它不执行 JavaScript，不绕过登录、付费墙或反爬；当静态 HTML 无法得到足够正文时，工具明确返回 `requires_browser=true`，由上层选择其他搜索结果或未来的浏览器工具。

```text
web_search
-> fetch_url 下载与 URL 安全校验
-> Trafilatura bare_extraction() 结构化提取
-> web_extract_quality_reviewer 审查正文可用性
-> 字符阈值判断
-> web_extract_llm 单次正文压缩
-> Agent 可消费的 content + extraction 元数据
```

## 2. 提取层

实现位于 `external_info/web_extraction.py`，使用：

```python
bare_extraction(
    html,
    url=url,
    include_comments=False,
    include_links=True,
    include_images=False,
    include_tables=True,
    favor_precision=True,
)
```

提取结果统一包含：`title`、`author`、`published_at`、`description`、`site_name`、`hostname`、`categories`、`tags`、`text`、`extractor`、`requires_browser`。`document is None` 或正文为空时，返回 `success=false` 与 `EXTRACTION_FAILED`，不把噪声 HTML 当成正文。

## 3. 质量审查

每个有正文的 HTML 抽取结果默认交给独立的 `web_extract_quality_reviewer`。它和主 Agent 共用 Chat Completions API 凭据，但使用单独的 system prompt，只输出以下 JSON：

```json
{
  "usable": true,
  "quality": "high",
  "reason": "正文连贯且包含事件信息。",
  "requires_alternate_url": false
}
```

审查输入包含标题、描述、站点、正文总字符数及最多 12,000 字符的正文预览，不把超长全文再送入审查调用。判为不可用时：

- `success=false`
- `content=""`
- `requires_browser=true`（或 `requires_alternate_url=true`）
- Agent 应优先切换搜索结果，而不是基于噪声作答。

若审查模型暂不可用，提取结果不会被静默丢弃：返回 `quality=unknown`、`review_error=true`，并保留正文让上层决定是否继续。这避免辅助模型故障直接中断联网能力。

## 4. 文本压缩策略

质量合格后按原始正文字符数处理：

| 原始正文字符数 | 策略 | 结果 |
| --- | --- | --- |
| `<= 5,000` | 不调用模型 | 原文作为 `content` 返回 |
| `5,001 - 500,000` | 一次 `web_extract_llm` 调用 | 尽量保留事实的压缩正文，默认最多 5,000 字符 |
| `> 500,000` | 不压缩 | 保留原文并标记 `source_exceeds_500000_chars`，由上层拆分或改用浏览器/站点适配器 |

压缩提示词要求保留事件主体、时间、人物和机构、数据、因果关系、不同观点、限定条件与不确定性；禁止补写网页中不存在的内容。`content` 始终是给 Agent 的正文，`extraction.compression` 记录源长度、输出长度、模型和失败原因。

## 5. fetch_url 协议

`fetch_url` 参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `quality_check` | `true` | 是否运行独立质量审查模型。 |
| `compress` | `true` | 是否在超过 5,000 字符时运行压缩模型。 |
| `max_bytes` | `1000000` | 下载字节上限，仍受 5 MB 硬上限保护。 |
| `timeout_ms` | `15000` | 下载超时，仍受 60 秒硬上限保护。 |

HTML 成功响应摘要：

```json
{
  "url": "https://example.com/article",
  "success": true,
  "requires_browser": false,
  "content": "给 Agent 的正文或压缩正文",
  "extraction": {
    "title": "...",
    "published_at": "...",
    "extractor": "trafilatura",
    "source_text_char_count": 8400,
    "quality": {"checked": true, "usable": true, "quality": "high"},
    "compression": {"applied": true, "method": "web_extract_llm", "output_char_count": 2800}
  }
}
```

原始 `text` 不会同时放进 `extraction`，以免压缩完成后又把超长原文重复塞回模型上下文。若未来确实需要下载 JSON API 或原始二进制内容，应新增语义明确的专用工具，而不是重新给 `fetch_url` 增加多模式分支。

## 6. 配置与依赖

依赖文件：`python_demo_test/agent_input_framework_mvp/requirements.txt`。

默认复用主 Agent 的 `AGENT_LLM_BASE_URL`、`AGENT_LLM_API_KEY`、`AGENT_LLM_API_KEY_FILE`、`AGENT_LLM_MODEL_NAME`。以下变量可为网页辅助模型单独覆盖：

| 变量 | 用途 |
| --- | --- |
| `AGENT_WEB_EXTRACT_BASE_URL` | 审查/压缩模型 API 地址。 |
| `AGENT_WEB_EXTRACT_API_KEY` | 审查/压缩模型密钥。 |
| `AGENT_WEB_EXTRACT_API_KEY_FILE` | 密钥文件，UTF-8。 |
| `AGENT_WEB_EXTRACT_MODEL_NAME` | 审查/压缩模型名。 |
| `AGENT_WEB_EXTRACT_TIMEOUT_SECONDS` | 单次模型请求超时，默认 90 秒。 |
| `AGENT_WEB_EXTRACT_USE_ENV_PROXY` | 是否使用系统代理，默认 `false`，与主 Agent 一致。 |
| `AGENT_WEB_EXTRACT_QUALITY_PREVIEW_CHARS` | 审查正文预览长度，默认 12,000。 |
| `AGENT_WEB_EXTRACT_SUMMARY_MAX_CHARS` | 压缩输出上限，默认 5,000。 |

## 7. 已知限制与后续

- Trafilatura 只能处理服务器返回的静态 HTML；前端渲染、反爬页和登录页应转浏览器工具。
- 单次压缩的可处理上限按字符数为 500,000，但仍受模型上下文窗口限制；生产环境应结合实际模型窗口设置更小上限，或实现分块压缩。
- 当前质量审查失败采用“保留正文并标记未知”的可用性优先策略；如果业务场景要求更严格，可将策略改为 fail-closed。
- 下一阶段可以加入站点适配器和浏览器工具，使 `requires_browser` 能自动触发后续行动。
