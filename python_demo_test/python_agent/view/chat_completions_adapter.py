from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .prompt import (
    AssistantMessage,
    ContentBlock,
    ContentBlockType,
    Message,
    MessageHistory,
    PromptRequest,
    Role,
    SystemPrompt,
    Tool as PromptTool,
    ToolCatalog,
    ToolResultMessage,
    ToolUse,
    UserMessage,
    to_plain_dict,
)
from .tooling import ApiToolSchema, ToolRegistry, ToolResult, ToolSpec


class ChatCompletionsAdapter:
    """Convert engineering-facing view objects into Chat Completions payloads.

    The adapter is deliberately provider-shaped at the boundary:
    - internal PromptRequest/MessageHistory/ToolCatalog stay rich and typed;
    - output is a plain dict compatible with OpenAI-style Chat Completions.
    """

    def to_chat_completions_request(self, prompt_request: PromptRequest) -> Dict[str, Any]:
        """Serialize PromptRequest into a Chat Completions request dict."""

        messages = self._system_messages(prompt_request.system)
        messages.extend(
            self.message_to_chat_message(message)
            for message in prompt_request.messages.messages
        )
        payload: Dict[str, Any] = {
            "model": prompt_request.model,
            "messages": messages,
            "stream": prompt_request.stream,
        }
        tools = self.tool_catalog_to_chat_tools(prompt_request.tools)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = prompt_request.tool_choice.value
        return payload

    def tool_registry_to_chat_tools(self, registry: ToolRegistry) -> List[Dict[str, Any]]:
        """Serialize ToolRegistry into Chat Completions function tool schemas."""

        return [
            self.api_tool_schema_to_chat_tool(tool.api_tool_schema())
            for tool in registry.list()
            if tool.model_visible() and not tool.defer_loading()
        ]

    def tool_catalog_to_chat_tools(self, catalog: ToolCatalog) -> List[Dict[str, Any]]:
        """Serialize prompt-layer ToolCatalog into Chat Completions tools."""

        return [
            self.prompt_tool_to_chat_tool(tool)
            for tool in catalog.active_tools
            if not tool.defer_loading
        ]

    def prompt_tool_to_chat_tool(self, tool: PromptTool) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
                "strict": tool.strict,
            },
        }

    def tool_spec_to_chat_tool(self, tool: ToolSpec) -> Dict[str, Any]:
        return self.api_tool_schema_to_chat_tool(tool.api_tool_schema())

    def api_tool_schema_to_chat_tool(self, schema: ApiToolSchema) -> Dict[str, Any]:
        return schema.to_provider_dict()

    def message_history_to_chat_messages(self, history: MessageHistory) -> List[Dict[str, Any]]:
        return [self.message_to_chat_message(message) for message in history.messages]

    def message_to_chat_message(self, message: Message) -> Dict[str, Any]:
        """Serialize one internal message into Chat Completions message format."""

        if isinstance(message, ToolResultMessage):
            return self.tool_result_message_to_chat_message(message)
        if isinstance(message, AssistantMessage):
            return self.assistant_message_to_chat_message(message)
        if isinstance(message, UserMessage):
            return self.user_message_to_chat_message(message)
        if message.role == Role.TOOL:
            return self._generic_tool_message_to_chat_message(message)
        return {
            "role": message.role.value,
            "content": self._content_blocks_to_text(message.content),
        }

    def user_message_to_chat_message(self, message: UserMessage) -> Dict[str, Any]:
        """Serialize user input plus turn metadata into the current user message."""

        raw_content = self._content_blocks_to_text(message.content) or message.raw_user_text
        metadata = self._non_empty_metadata(to_plain_dict(message.turn_metadata))
        if not metadata:
            return {
                "role": Role.USER.value,
                "content": raw_content,
            }
        return {
            "role": Role.USER.value,
            "content": "\n\n".join(
                [
                    raw_content,
                    "Turn Metadata:",
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                ]
            ),
        }

    def assistant_message_to_chat_message(self, message: AssistantMessage) -> Dict[str, Any]:
        text = message.text
        if text is None:
            text = self._content_blocks_to_text(
                block for block in message.content if block.block_type == ContentBlockType.TEXT
            )
        payload: Dict[str, Any] = {
            "role": Role.ASSISTANT.value,
            "content": text if text else None,
        }
        tool_uses = self._dedupe_tool_uses(
            list(message.tool_uses)
            + [
                block.tool_use
                for block in message.content
                if block.block_type == ContentBlockType.TOOL_USE and block.tool_use is not None
            ]
        )
        if tool_uses:
            payload["tool_calls"] = [
                self.tool_use_to_chat_tool_call(tool_use, index)
                for index, tool_use in enumerate(tool_uses)
            ]
        return payload

    def tool_use_to_chat_tool_call(self, tool_use: ToolUse, index: int = 0) -> Dict[str, Any]:
        tool_call_id = tool_use.tool_use_id or f"call_{index}"
        return {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_use.name,
                "arguments": json.dumps(tool_use.input or {}, ensure_ascii=False),
            },
        }

    def tool_result_message_to_chat_message(self, message: ToolResultMessage) -> Dict[str, Any]:
        return {
            "role": Role.TOOL.value,
            "tool_call_id": message.tool_use_id,
            "content": message.result_content or self._content_blocks_to_text(message.content),
        }

    def tool_result_to_chat_message(self, tool_call_id: str, result: ToolResult) -> Dict[str, Any]:
        return {
            "role": Role.TOOL.value,
            "tool_call_id": tool_call_id,
            "content": result.content,
        }

    def assistant_tool_calls_to_chat_message(
        self,
        tool_uses: Iterable[ToolUse],
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "role": Role.ASSISTANT.value,
            "content": content,
            "tool_calls": [
                self.tool_use_to_chat_tool_call(tool_use, index)
                for index, tool_use in enumerate(tool_uses)
            ],
        }

    def assistant_chat_message_to_internal(self, message: Dict[str, Any]) -> AssistantMessage:
        """Parse a Chat Completions assistant message into internal AssistantMessage."""

        tool_uses: List[ToolUse] = []
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments_text = function.get("arguments") or "{}"
            try:
                parsed_input = json.loads(arguments_text)
                if not isinstance(parsed_input, dict):
                    parsed_input = {"_value": parsed_input}
            except json.JSONDecodeError as exc:
                parsed_input = {
                    "_raw_arguments": arguments_text,
                    "_parse_error": exc.msg,
                }
            tool_uses.append(
                ToolUse(
                    tool_use_id=str(tool_call.get("id", "")),
                    name=str(function.get("name", "")),
                    input=parsed_input,
                )
            )
        return AssistantMessage(
            text=message.get("content"),
            tool_uses=tool_uses,
        )

    def _dedupe_tool_uses(self, tool_uses: Iterable[ToolUse]) -> List[ToolUse]:
        result: List[ToolUse] = []
        seen: set[str] = set()
        for index, tool_use in enumerate(tool_uses):
            key = tool_use.tool_use_id or f"{tool_use.name}:{json.dumps(tool_use.input, sort_keys=True, ensure_ascii=False)}:{index}"
            if key in seen:
                continue
            seen.add(key)
            result.append(tool_use)
        return result

    def _system_messages(self, system: SystemPrompt) -> List[Dict[str, Any]]:
        rendered = system.render()
        if not rendered:
            return []
        return [{"role": Role.SYSTEM.value, "content": rendered}]

    def _non_empty_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        empty_values = ("", None, [], {})
        return {
            key: value
            for key, value in metadata.items()
            if value not in empty_values
        }

    def _generic_tool_message_to_chat_message(self, message: Message) -> Dict[str, Any]:
        tool_call_id = ""
        for block in message.content:
            if block.tool_use_id:
                tool_call_id = block.tool_use_id
                break
        return {
            "role": Role.TOOL.value,
            "tool_call_id": tool_call_id,
            "content": self._content_blocks_to_text(message.content),
        }

    def _content_blocks_to_text(self, blocks: Iterable[ContentBlock]) -> str:
        parts: List[str] = []
        for block in blocks:
            if block.block_type in (ContentBlockType.TEXT, ContentBlockType.TOOL_RESULT):
                parts.append(block.text)
            elif block.block_type == ContentBlockType.THINKING and block.thinking:
                parts.append(block.thinking.text)
            elif block.block_type == ContentBlockType.TOOL_USE and block.tool_use:
                parts.append(json.dumps(self.tool_use_to_chat_tool_call(block.tool_use), ensure_ascii=False))
            elif block.block_type == ContentBlockType.IMAGE_URL and block.image_url:
                parts.append(block.image_url)
            elif block.block_type == ContentBlockType.SERVER_TOOL_USE and block.server_tool_name:
                parts.append(block.server_tool_name)
        return "\n".join(part for part in parts if part)


def build_chat_completions_adapter_demo() -> Dict[str, Any]:
    """Show the engineering view -> Chat Completions request/tool-result flow."""

    from .prompt import MessageHistory, PromptRequest, SystemPrompt, Tool as PromptTool, ToolCatalog, TurnMetadata
    from .tooling import ToolContext, build_demo_tool_registry

    registry = build_demo_tool_registry()
    adapter = ChatCompletionsAdapter()
    prompt_tools = [
        PromptTool(
            name=tool.name(),
            description=tool.description(),
            input_schema=tool.input_schema(),
            strict=False,
            defer_loading=tool.defer_loading(),
        )
        for tool in registry.list()
    ]
    history = MessageHistory()
    history.append_user_message(
        "请调用 echo 工具返回 hello chat completions",
        turn_metadata=TurnMetadata(
            current_date="2026-07-08",
            workspace="demo-workspace",
            model="demo-chat-model",
            mode="adapter_demo",
            input_provenance="demo",
            working_set_summary="Chat Completions adapter demo.",
        ),
    )
    prompt_request = PromptRequest(
        model="demo-chat-model",
        system=SystemPrompt(
            stable_prefix="你是一个会按需调用工具的 demo agent。",
            runtime_instructions="如果需要 echo，请使用工具调用。",
        ),
        messages=history,
        tools=ToolCatalog(active_tools=prompt_tools),
        stream=True,
    )
    chat_request = adapter.to_chat_completions_request(prompt_request)

    simulated_assistant = adapter.assistant_tool_calls_to_chat_message(
        [
            ToolUse(
                tool_use_id="call_demo_echo",
                name="echo",
                input={"text": "hello chat completions"},
            )
        ],
        content=None,
    )
    internal_assistant = adapter.assistant_chat_message_to_internal(simulated_assistant)
    tool_use = internal_assistant.tool_uses[0]
    tool_result = registry.execute_full_with_context(
        name=tool_use.name,
        input=tool_use.input,
        ctx=ToolContext(workspace_root="demo-workspace", session_id="session_demo", turn_id="turn_demo"),
    )
    tool_message = adapter.tool_result_to_chat_message(tool_use.tool_use_id, tool_result)

    return {
        "chat_completions_request": chat_request,
        "simulated_assistant_tool_call_message": simulated_assistant,
        "parsed_internal_tool_use": {
            "tool_use_id": tool_use.tool_use_id,
            "name": tool_use.name,
            "input": tool_use.input,
        },
        "tool_result_message": tool_message,
    }
