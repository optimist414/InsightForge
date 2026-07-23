from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


def commented_field(
    comment: str,
    default: Any = None,
    default_factory: Any = None,
) -> Any:
    """Create a dataclass field with a machine-readable Chinese comment."""

    metadata = {"comment": comment}
    if default_factory is not None:
        return field(default_factory=default_factory, metadata=metadata)
    return field(default=default, metadata=metadata)


def to_plain_dict(value: Any) -> Any:
    """Convert prompt dataclasses/enums into JSON-serializable values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    return value


class Role(str, Enum):
    """Provider-compatible message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentBlockType(str, Enum):
    """Block types supported by the internal prompt view."""

    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE_URL = "image_url"
    SERVER_TOOL_USE = "server_tool_use"


class ToolChoiceMode(str, Enum):
    """Tool-choice modes before converting to a specific provider request."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass
class ThinkingConfig:
    """Optional provider thinking configuration."""

    enabled: bool = commented_field("是否启用模型思考配置。", False)
    budget_tokens: Optional[int] = commented_field("思考 token 预算；None 表示不指定。", None)


@dataclass
class Thinking:
    """Assistant thinking block stored in the internal view when a provider returns it."""

    text: str = commented_field("模型返回的思考文本或摘要。", "")
    redacted: bool = commented_field("是否已脱敏或隐藏。", False)


@dataclass
class ToolUse:
    """One assistant tool-use request."""

    tool_use_id: str = commented_field("工具调用 ID，用于和 ToolResultMessage 对齐。", "")
    name: str = commented_field("工具名称。", "")
    input: Dict[str, Any] = commented_field("工具输入参数。", default_factory=dict)


@dataclass
class ContentBlock:
    """A typed message content block.

    Provider adapters can flatten text-only providers, or preserve richer
    blocks for providers that support tool use, images, or server tools.
    """

    block_type: ContentBlockType = commented_field("内容块类型。", ContentBlockType.TEXT)
    text: str = commented_field("文本内容；Text/Thinking/ToolResult 常用。", "")
    thinking: Optional[Thinking] = commented_field("思考内容块；当前 MVP 默认不发送给 provider。", None)
    tool_use: Optional[ToolUse] = commented_field("工具调用内容块。", None)
    tool_use_id: str = commented_field("工具结果对应的 tool_use_id。", "")
    image_url: str = commented_field("图片 URL；为多模态 provider 预留。", "")
    server_tool_name: str = commented_field("服务端工具名称；为 provider 内置工具预留。", "")
    is_error: bool = commented_field("工具结果是否为错误。", False)

    @classmethod
    def text_block(cls, text: str) -> "ContentBlock":
        return cls(block_type=ContentBlockType.TEXT, text=text)

    @classmethod
    def tool_result_block(cls, tool_use_id: str, content: str, is_error: bool = False) -> "ContentBlock":
        return cls(
            block_type=ContentBlockType.TOOL_RESULT,
            text=content,
            tool_use_id=tool_use_id,
            is_error=is_error,
        )


@dataclass
class Message:
    """Internal message object used by PromptRequest."""

    role: Role = commented_field("消息角色：system/user/assistant/tool。", Role.USER)
    content: List[ContentBlock] = commented_field("消息内容块列表。", default_factory=list)

    def to_provider_content(self) -> str:
        """Flatten rich content blocks into text for a chat-completions provider."""

        rendered: List[str] = []
        for block in self.content:
            if block.block_type in (ContentBlockType.TEXT, ContentBlockType.TOOL_RESULT):
                rendered.append(block.text)
            elif block.block_type == ContentBlockType.TOOL_USE and block.tool_use:
                rendered.append(json.dumps(to_plain_dict(block.tool_use), ensure_ascii=False))
            elif block.block_type == ContentBlockType.THINKING and block.thinking:
                rendered.append(block.thinking.text)
            elif block.block_type == ContentBlockType.IMAGE_URL and block.image_url:
                rendered.append(f"[image_url] {block.image_url}")
            elif block.block_type == ContentBlockType.SERVER_TOOL_USE and block.server_tool_name:
                rendered.append(f"[server_tool_use] {block.server_tool_name}")
        return "\n".join(part for part in rendered if part)


@dataclass
class TurnMetadata:
    """Metadata snapshot attached to the current user turn."""

    current_date: str = commented_field("当前日期，用于处理相对日期和时间敏感任务。", "")
    workspace: str = commented_field("当前工作区路径或项目标识。", "")
    model: str = commented_field("本轮计划调用的模型名称。", "")
    mode: str = commented_field("当前运行模式，例如 rule/deepseek/plan/default。", "")
    permissions: Dict[str, Any] = commented_field("当前权限和沙箱信息摘要。", default_factory=dict)
    input_provenance: str = commented_field("用户输入来源，例如 chat/cli/api。", "chat")
    resource_metadata: Dict[str, Any] = commented_field("相关文件、MCP、skill、数据源等资源元数据。", default_factory=dict)
    working_set_summary: str = commented_field("当前工作集摘要，避免把大量文件直接塞入 prompt。", "")


@dataclass
class UserMessage(Message):
    """User message with raw text and turn metadata."""

    raw_user_text: str = commented_field("用户原始文本，不做改写。", "")
    turn_metadata: TurnMetadata = commented_field("本轮输入对应的环境快照。", default_factory=TurnMetadata)

    def __post_init__(self) -> None:
        self.role = Role.USER
        if not self.content and self.raw_user_text:
            self.content = [ContentBlock.text_block(self.raw_user_text)]


@dataclass
class AssistantMessage(Message):
    """Assistant message with optional text, thinking and tool calls."""

    text: Optional[str] = commented_field("assistant 可展示文本。", None)
    thinking: Optional[Thinking] = commented_field("assistant 思考摘要或 provider thinking。", None)
    tool_uses: List[ToolUse] = commented_field("assistant 请求的工具调用列表。", default_factory=list)

    def __post_init__(self) -> None:
        self.role = Role.ASSISTANT
        if not self.content:
            blocks: List[ContentBlock] = []
            if self.text:
                blocks.append(ContentBlock.text_block(self.text))
            if self.thinking:
                blocks.append(ContentBlock(block_type=ContentBlockType.THINKING, thinking=self.thinking))
            for tool_use in self.tool_uses:
                blocks.append(ContentBlock(block_type=ContentBlockType.TOOL_USE, tool_use=tool_use))
            self.content = blocks


@dataclass
class ToolResultMessage(Message):
    """Tool result message linked to an assistant tool use."""

    tool_use_id: str = commented_field("对应 ToolUse.tool_use_id。", "")
    result_content: str = commented_field("工具返回内容，provider 不支持结构时会转成文本。", "")
    is_error: bool = commented_field("工具结果是否表示错误。", False)

    def __post_init__(self) -> None:
        self.role = Role.TOOL
        if not self.content:
            self.content = [
                ContentBlock.tool_result_block(
                    tool_use_id=self.tool_use_id,
                    content=self.result_content,
                    is_error=self.is_error,
                )
            ]


@dataclass
class MessageHistory:
    """Mutable prompt-facing message history."""

    messages: List[Message] = commented_field("按 provider 发送顺序保存的消息列表。", default_factory=list)

    def append_user_message(self, raw_user_text: str, turn_metadata: Optional[TurnMetadata] = None) -> None:
        self.messages.append(
            UserMessage(
                raw_user_text=raw_user_text,
                turn_metadata=turn_metadata or TurnMetadata(),
            )
        )

    def append_assistant_message(
        self,
        text: Optional[str] = None,
        thinking: Optional[Thinking] = None,
        tool_uses: Optional[List[ToolUse]] = None,
    ) -> None:
        self.messages.append(
            AssistantMessage(
                text=text,
                thinking=thinking,
                tool_uses=tool_uses or [],
            )
        )

    def append_tool_result(self, tool_use_id: str, content: str, is_error: bool = False) -> None:
        self.messages.append(
            ToolResultMessage(
                tool_use_id=tool_use_id,
                result_content=content,
                is_error=is_error,
            )
        )

    def messages_with_turn_metadata(self) -> List[Dict[str, Any]]:
        """Return serializable messages plus user-turn metadata."""

        rows: List[Dict[str, Any]] = []
        for message in self.messages:
            row = {
                "role": message.role.value,
                "content": [to_plain_dict(block) for block in message.content],
            }
            if isinstance(message, UserMessage):
                row["raw_user_text"] = message.raw_user_text
                row["turn_metadata"] = to_plain_dict(message.turn_metadata)
            rows.append(row)
        return rows


@dataclass
class SystemPrompt:
    """System prompt split into stable, dynamic, and runtime parts."""

    stable_prefix: str = commented_field("稳定前缀：agent 身份、输出协议、安全边界等长期规则。", "")
    dynamic_context: str = commented_field("动态上下文：目标、loop 摘要、记忆、工具目录、外部参考等。", "")
    runtime_instructions: str = commented_field("运行时指令：本轮 mode、权限、输出格式、工具调用限制等。", "")

    def render(self) -> str:
        parts = [
            ("Stable Prefix", self.stable_prefix),
            ("Dynamic Context", self.dynamic_context),
            ("Runtime Instructions", self.runtime_instructions),
        ]
        return "\n\n".join(f"## {title}\n{content}" for title, content in parts if content.strip())


@dataclass
class Tool:
    """Provider-visible tool declaration."""

    name: str = commented_field("工具名称，必须和执行层注册名一致。", "")
    description: str = commented_field("工具用途说明，帮助模型判断何时调用。", "")
    input_schema: Dict[str, Any] = commented_field("JSON Schema 风格的输入参数定义。", default_factory=dict)
    strict: bool = commented_field("是否要求 provider 严格遵循 schema。", False)
    defer_loading: bool = commented_field("是否延迟加载工具详情；用于大工具目录。", False)

    @classmethod
    def from_registry_item(cls, item: Dict[str, Any]) -> "Tool":
        return cls(
            name=str(item.get("name", "")),
            description=str(item.get("description") or item.get("when_to_use") or ""),
            input_schema=item.get("input_schema") or {},
            strict=bool(item.get("strict", False)),
            defer_loading=bool(item.get("defer_loading", False)),
        )


@dataclass
class ToolCatalog:
    """Prompt-facing active tool catalog."""

    active_tools: List[Tool] = commented_field("本轮真正暴露给模型的工具列表。", default_factory=list)
    tool_choice: ToolChoiceMode = commented_field("工具选择策略。", ToolChoiceMode.AUTO)

    @classmethod
    def from_registry_description(cls, tool_catalog: Dict[str, Any]) -> "ToolCatalog":
        tools = tool_catalog.get("tools", tool_catalog)
        if isinstance(tools, dict):
            active_tools = [Tool.from_registry_item(value) for value in tools.values()]
        elif isinstance(tools, list):
            active_tools = [Tool.from_registry_item(value) for value in tools if isinstance(value, dict)]
        else:
            active_tools = []
        return cls(active_tools=active_tools)


@dataclass
class ProviderChatRequest:
    """Provider-adapted chat request."""

    model: str = commented_field("provider 模型名称。", "")
    messages: List[Dict[str, Any]] = commented_field("provider chat messages。", default_factory=list)
    tools: List[Dict[str, Any]] = commented_field("provider 工具声明；不支持时可为空。", default_factory=list)
    tool_choice: Any = commented_field("provider tool_choice 字段。", "auto")
    stream: bool = commented_field("是否流式返回。", False)


@dataclass
class PromptRequest:
    """Internal standard object for one LLM call."""

    model: str = commented_field("内部计划使用的模型名称。", "")
    system: SystemPrompt = commented_field("结构化 system prompt。", default_factory=SystemPrompt)
    messages: MessageHistory = commented_field("对话历史和当前用户消息。", default_factory=MessageHistory)
    tools: ToolCatalog = commented_field("本轮可用工具目录。", default_factory=ToolCatalog)
    tool_choice: ToolChoiceMode = commented_field("工具选择策略。", ToolChoiceMode.AUTO)
    stream: bool = commented_field("是否请求流式响应。", False)
    thinking: Optional[ThinkingConfig] = commented_field("可选 thinking 配置。", None)

    def to_provider_chat_request(self) -> ProviderChatRequest:
        provider_messages = [{"role": Role.SYSTEM.value, "content": self.system.render()}]
        provider_messages.extend(
            {
                "role": message.role.value,
                "content": message.to_provider_content(),
            }
            for message in self.messages.messages
        )
        return ProviderChatRequest(
            model=self.model,
            messages=provider_messages,
            tools=[self._tool_to_provider(tool) for tool in self.tools.active_tools if not tool.defer_loading],
            tool_choice=self.tool_choice.value,
            stream=self.stream,
        )

    def _tool_to_provider(self, tool: Tool) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
                "strict": tool.strict,
            },
        }


def prompt_field_comments() -> Dict[str, Dict[str, str]]:
    """Expose field comments for docs, schema generation, or debug output."""

    classes = [
        PromptRequest,
        SystemPrompt,
        MessageHistory,
        Message,
        UserMessage,
        AssistantMessage,
        ToolResultMessage,
        ContentBlock,
        TurnMetadata,
        ToolCatalog,
        Tool,
        ProviderChatRequest,
    ]
    result: Dict[str, Dict[str, str]] = {}
    for cls in classes:
        result[cls.__name__] = {
            name: field_info.metadata.get("comment", "")
            for name, field_info in cls.__dataclass_fields__.items()
        }
    return result
