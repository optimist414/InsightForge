from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional
from uuid import uuid4

from ..view.chat_completions_adapter import ChatCompletionsAdapter
from ..view.prompt import (
    AssistantMessage,
    Message,
    MessageHistory,
    PromptRequest,
    SystemPrompt,
    Tool as PromptTool,
    ToolResultMessage,
    ToolUse,
    TurnMetadata,
    to_plain_dict,
)
from ..view.tooling import (
    ToolContext,
    ToolFailureCode,
    ToolFailureDTO,
    ToolRegistry,
    ToolResult,
    build_demo_tool_registry,
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class StreamEventType(str, Enum):
    MESSAGE_START = "MessageStart"
    CONTENT_BLOCK_START = "ContentBlockStart"
    CONTENT_BLOCK_DELTA = "ContentBlockDelta"
    CONTENT_BLOCK_STOP = "ContentBlockStop"
    MESSAGE_DELTA = "MessageDelta"
    THINKING_DELTA = "ThinkingDelta"
    MESSAGE_STOP = "MessageStop"
    ERROR = "Error"


class CoreEventType(str, Enum):
    TURN_STARTED = "TurnStarted"
    MESSAGE_STARTED = "MessageStarted"
    MESSAGE_DELTA = "MessageDelta"
    MESSAGE_COMPLETE = "MessageComplete"
    THINKING_DELTA = "ThinkingDelta"
    TOOL_CALL_STARTED = "ToolCallStarted"
    TOOL_CALL_COMPLETE = "ToolCallComplete"
    TOOL_CALL_FAILED = "ToolCallFailed"
    TURN_COMPLETE = "TurnComplete"
    ERROR = "Error"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens


@dataclass
class Turn:
    id: str = field(default_factory=lambda: new_id("turn"))
    step: int = 0
    usage: Usage = field(default_factory=Usage)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def add_usage(self, usage: Usage) -> None:
        self.usage.add(usage)

    def record_tool_call(self, tool_name: str, tool_use_id: str, input: Dict[str, Any]) -> None:
        self.tool_calls.append(
            {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "input": input,
                "step": self.step,
            }
        )


@dataclass
class Session:
    messages: MessageHistory = field(default_factory=MessageHistory)
    system_prompt: str = ""
    working_set: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    event_type: StreamEventType
    content: str = ""
    block_type: str = ""
    tool_use_id: str = ""
    tool_name: str = ""
    tool_call_index: Optional[int] = None
    delta: str = ""
    usage: Optional[Usage] = None
    error: str = ""


@dataclass
class ToolUseState:
    id: str
    name: str
    input_buffer: str = ""
    parsed_input: Optional[Dict[str, Any]] = None
    caller: str = "assistant"
    input_parse_error: Optional[str] = None

    def append_delta(self, delta: str) -> None:
        self.input_buffer += delta

    def finalize(self) -> None:
        try:
            value = json.loads(self.input_buffer or "{}")
            self.parsed_input = value if isinstance(value, dict) else {"_value": value}
        except json.JSONDecodeError as exc:
            self.parsed_input = {}
            self.input_parse_error = exc.msg


@dataclass
class ToolExecutionPlan:
    tool_name: str
    input: Dict[str, Any]
    approval_required: bool = False
    read_only: bool = True
    parallel_group: Optional[str] = None
    tool_use_id: str = ""


@dataclass
class ToolExecutionOutcome:
    tool_use_id: str
    result: ToolResult
    duration_ms: int
    is_error: bool


@dataclass
class CoreEvent:
    event_type: CoreEventType
    payload: Dict[str, Any] = field(default_factory=dict)


class McpPool:
    def call_tool(self, server: str, tool: str, input: Dict[str, Any]) -> ToolResult:
        return ToolFailureDTO(
            tool_name=f"{server}.{tool}",
            error_code=ToolFailureCode.MCP_ERROR,
            message=f"MCP call is not wired: {server}.{tool}",
            retryable=False,
            failed_stage="dependency",
            suggested_action="配置对应 MCP server 后重试，或选择其他工具。",
            cause_type="mcp_not_configured",
        ).to_result()


class LlmClient:
    def create_message_stream(self, request: Dict[str, Any]) -> Iterable[StreamEvent]:
        raise NotImplementedError


class ScriptedLlmClient(LlmClient):
    """Demo LLM client.

    Step 1 returns a streaming tool call. After the tool result appears in
    messages, step 2 returns a final assistant text.
    """

    def create_message_stream(self, request: Dict[str, Any]) -> Iterable[StreamEvent]:
        messages = request.get("messages", [])
        last_user_index = max(
            (index for index, message in enumerate(messages) if message.get("role") == "user"),
            default=-1,
        )
        current_turn_messages = messages[last_user_index + 1 :] if last_user_index >= 0 else messages
        has_tool_result = any(message.get("role") == "tool" for message in current_turn_messages)
        turn_number = sum(1 for message in messages if message.get("role") == "user")
        tool_use_id = f"call_demo_echo_{turn_number}"
        tool_text = f"hello from engine loop turn {turn_number}"
        latest_tool_content = next(
            (
                str(message.get("content", ""))
                for message in reversed(current_turn_messages)
                if message.get("role") == "tool"
            ),
            tool_text,
        )
        yield StreamEvent(StreamEventType.MESSAGE_START)
        if not has_tool_result:
            yield StreamEvent(
                StreamEventType.CONTENT_BLOCK_START,
                block_type="tool_use",
                tool_use_id=tool_use_id,
                tool_name="echo",
            )
            yield StreamEvent(StreamEventType.CONTENT_BLOCK_DELTA, delta='{"text": ')
            yield StreamEvent(StreamEventType.CONTENT_BLOCK_DELTA, delta=json.dumps(tool_text, ensure_ascii=False))
            yield StreamEvent(StreamEventType.CONTENT_BLOCK_DELTA, delta="}")
            yield StreamEvent(StreamEventType.CONTENT_BLOCK_STOP)
        else:
            yield StreamEvent(StreamEventType.CONTENT_BLOCK_START, block_type="text")
            yield StreamEvent(StreamEventType.CONTENT_BLOCK_DELTA, delta=f"工具已返回：{latest_tool_content}")
            yield StreamEvent(StreamEventType.CONTENT_BLOCK_STOP, block_type="text")
        yield StreamEvent(StreamEventType.MESSAGE_STOP, usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2))


@dataclass
class MessageRequestBuilder:
    session: Session
    tool_registry: ToolRegistry
    adapter: ChatCompletionsAdapter = field(default_factory=ChatCompletionsAdapter)
    _tool_choice: str = "auto"
    _stream: bool = True
    _runtime_instructions: str = ""

    def messages_with_turn_metadata(self) -> MessageHistory:
        return self.session.messages

    def active_tools(self) -> List[PromptTool]:
        return [
            PromptTool(
                name=tool.name(),
                description=tool.description(),
                input_schema=tool.input_schema(),
                strict=False,
                defer_loading=tool.defer_loading(),
            )
            for tool in self.tool_registry.list()
            if tool.model_visible()
        ]

    def tool_choice(self, value: str = "auto") -> "MessageRequestBuilder":
        self._tool_choice = value
        return self

    def stream_true(self) -> "MessageRequestBuilder":
        self._stream = True
        return self

    def runtime_instructions(self, value: str) -> "MessageRequestBuilder":
        self._runtime_instructions = value.strip()
        return self

    def stable_prefix_with_tool_directory(self, active_tools: List[PromptTool]) -> str:
        base_prompt = self.session.system_prompt.strip()
        if not active_tools:
            tool_directory = "当前步骤不开放工具调用，请直接根据已有上下文回答。"
        else:
            directory = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in active_tools
            ]
            tool_directory = (
                "当前可用工具目录如下。需要调用工具时必须使用 Chat Completions tools 协议，"
                "不要在正文中伪造工具调用。\n"
                + json.dumps(directory, ensure_ascii=False, indent=2)
            )
        return "\n\n".join(part for part in (base_prompt, tool_directory) if part)

    def build(self) -> Dict[str, Any]:
        from ..view.prompt import ToolCatalog, ToolChoiceMode
        active_tools = [] if self._tool_choice == "none" else self.active_tools()

        prompt_request = PromptRequest(
            model=str(self.session.settings.get("model", "demo-chat-model")),
            system=SystemPrompt(
                stable_prefix=self.stable_prefix_with_tool_directory(active_tools),
                runtime_instructions=self._runtime_instructions,
            ),
            messages=self.messages_with_turn_metadata(),
            tools=ToolCatalog(active_tools=active_tools),
            tool_choice=ToolChoiceMode(self._tool_choice),
            stream=self._stream,
        )
        return self.adapter.to_chat_completions_request(prompt_request)


class StopController:
    def should_stop(self, turn: Turn, session: Session, events: List[CoreEvent]) -> bool:
        raise NotImplementedError

    def should_force_final_summary(self, turn: Turn, session: Session, events: List[CoreEvent]) -> bool:
        return False


@dataclass
class MaxStepsStopController(StopController):
    max_steps: int = 20

    def should_stop(self, turn: Turn, session: Session, events: List[CoreEvent]) -> bool:
        return turn.step >= self.max_steps

    def should_force_final_summary(self, turn: Turn, session: Session, events: List[CoreEvent]) -> bool:
        return turn.step == self.max_steps


@dataclass
class Engine:
    session: Session
    llm_client: LlmClient
    tool_registry: ToolRegistry
    mcp_pool: McpPool = field(default_factory=McpPool)
    stop_controller: StopController = field(default_factory=lambda: MaxStepsStopController(max_steps=20))
    adapter: ChatCompletionsAdapter = field(default_factory=ChatCompletionsAdapter)
    core_events: List[CoreEvent] = field(default_factory=list)
    event_listener: Optional[Callable[[CoreEvent], None]] = None

    def handle_send_message(self, user_text: str) -> List[CoreEvent]:
        turn = Turn()
        self.emit_event(CoreEvent(CoreEventType.TURN_STARTED, {"turn_id": turn.id}))
        self.add_session_message(
            "user",
            user_text,
            turn_metadata=TurnMetadata(
                current_date=str(self.session.settings.get("current_date", "")),
                workspace=str(self.session.working_set.get("workspace_root", "")),
                model=str(self.session.settings.get("model", "demo-chat-model")),
                mode="engine_demo",
                input_provenance="engine",
            ),
        )
        self.refresh_system_prompt()
        self.run_turn_loop(turn)
        self.emit_event(
            CoreEvent(
                CoreEventType.TURN_COMPLETE,
                {"turn_id": turn.id, "steps": turn.step, "usage": to_plain_dict(turn.usage)},
            )
        )
        return self.core_events

    def refresh_system_prompt(self) -> None:
        if not self.session.system_prompt:
            self.session.system_prompt = "你是一个支持 Chat Completions 工具调用协议的 demo agent。"

    def run_turn_loop(self, turn: Turn) -> None:
        while not self.stop_controller.should_stop(turn, self.session, self.core_events):
            turn.next_step()
            force_final_summary = self.stop_controller.should_force_final_summary(
                turn, self.session, self.core_events
            )
            request = (
                MessageRequestBuilder(self.session, self.tool_registry, self.adapter)
                .tool_choice("none" if force_final_summary else "auto")
                .runtime_instructions(
                    self._forced_final_summary_prompt() if force_final_summary else ""
                )
                .stream_true()
                .build()
            )
            stream = self.llm_client.create_message_stream(request)
            assistant_text_parts: List[str] = []
            tool_states: List[ToolUseState] = []
            tool_states_by_index: Dict[int, ToolUseState] = {}
            active_tool_state: Optional[ToolUseState] = None
            self.emit_event(
                CoreEvent(
                    CoreEventType.MESSAGE_STARTED,
                    {"step": turn.step, "force_final_summary": force_final_summary},
                )
            )

            for event in stream:
                if event.event_type == StreamEventType.CONTENT_BLOCK_START and event.block_type == "tool_use":
                    tool_state = ToolUseState(id=event.tool_use_id, name=event.tool_name)
                    active_tool_state = tool_state
                    tool_states.append(tool_state)
                    if event.tool_call_index is not None:
                        tool_states_by_index[event.tool_call_index] = tool_state
                    self.emit_event(
                        CoreEvent(
                            CoreEventType.TOOL_CALL_STARTED,
                            {"tool_use_id": event.tool_use_id, "tool_name": event.tool_name},
                        )
                    )
                elif event.event_type == StreamEventType.CONTENT_BLOCK_DELTA:
                    target_tool_state = (
                        tool_states_by_index.get(event.tool_call_index)
                        if event.tool_call_index is not None
                        else active_tool_state
                    )
                    if event.block_type == "tool_use" or target_tool_state is not None:
                        if target_tool_state is not None:
                            target_tool_state.append_delta(event.delta)
                    else:
                        assistant_text_parts.append(event.delta)
                        self.emit_event(CoreEvent(CoreEventType.MESSAGE_DELTA, {"delta": event.delta}))
                elif event.event_type == StreamEventType.THINKING_DELTA:
                    self.emit_event(CoreEvent(CoreEventType.THINKING_DELTA, {"delta": event.delta}))
                elif event.event_type == StreamEventType.CONTENT_BLOCK_STOP:
                    target_tool_state = (
                        tool_states_by_index.get(event.tool_call_index)
                        if event.tool_call_index is not None
                        else active_tool_state
                    )
                    if event.block_type == "tool_use" or target_tool_state is not None:
                        if target_tool_state is not None:
                            target_tool_state.finalize()
                        if event.tool_call_index is not None:
                            tool_states_by_index.pop(event.tool_call_index, None)
                        active_tool_state = None
                elif event.event_type == StreamEventType.MESSAGE_STOP:
                    if event.usage:
                        turn.add_usage(event.usage)
                elif event.event_type == StreamEventType.ERROR:
                    self.emit_event(CoreEvent(CoreEventType.ERROR, {"error": event.error}))
                    return

            if tool_states:
                assistant_message = AssistantMessage(
                    text=None,
                    tool_uses=[
                        ToolUse(
                            tool_use_id=state.id,
                            name=state.name,
                            input=state.parsed_input or {},
                        )
                        for state in tool_states
                    ],
                )
                self.session.messages.messages.append(assistant_message)
                pending_skill_contexts: List[str] = []
                for state in tool_states:
                    plan = ToolExecutionPlan(
                        tool_name=state.name,
                        input=state.parsed_input or {},
                        approval_required=False,
                        read_only=True,
                        tool_use_id=state.id,
                    )
                    turn.record_tool_call(plan.tool_name, plan.tool_use_id, plan.input)
                    outcome = self.execute_tool_with_lock(plan, turn)
                    self.session.messages.append_tool_result(
                        tool_use_id=outcome.tool_use_id,
                        content=outcome.result.content,
                        is_error=outcome.is_error,
                    )
                    # 动态 skill 正文作为 assistant 上下文发送，避免固定占用 system prompt。
                    # 必须等同一 assistant 消息的全部 tool_call_id 都收到 tool 结果后再写入，
                    # 否则会破坏 Chat Completions 的 tool_calls / tool 消息连续性约束。
                    if plan.tool_name == "skill_loader" and not outcome.is_error:
                        metadata = outcome.result.metadata or {}
                        assistant_skill = metadata.get("assistant_message")
                        if assistant_skill:
                            pending_skill_contexts.append(str(assistant_skill))
                for assistant_skill in pending_skill_contexts:
                    self.session.messages.append_assistant_message(text=assistant_skill)
                continue

            final_text = "".join(assistant_text_parts)
            if self._contains_dsml(final_text) and not self.stop_controller.should_stop(
                turn, self.session, self.core_events
            ):
                final_text = self.recover_dsml_final(turn, final_text)
            elif self._contains_dsml(final_text):
                final_text = self._fallback_summary_from_thinking()
            self.add_session_message("assistant", final_text)
            self.emit_event(CoreEvent(CoreEventType.MESSAGE_COMPLETE, {"content": final_text}))
            return

    def recover_dsml_final(self, turn: Turn, leaked_text: str) -> str:
        self.add_session_message("assistant", leaked_text)
        self.add_session_message(
            "user",
            "工具调用阶段已经结束。禁止继续调用任何工具，也不要输出 DSML、XML、SQL 或工具协议。"
            "请只根据已有工具结果，直接给出简洁、完整的中文最终回答。",
        )
        turn.next_step()
        self.emit_event(CoreEvent(CoreEventType.MESSAGE_STARTED, {"step": turn.step}))
        request = (
            MessageRequestBuilder(self.session, self.tool_registry, self.adapter)
            .tool_choice("none")
            .stream_true()
            .build()
        )
        text_parts: List[str] = []
        for event in self.llm_client.create_message_stream(request):
            if event.event_type == StreamEventType.CONTENT_BLOCK_DELTA and event.block_type != "tool_use":
                text_parts.append(event.delta)
                self.emit_event(CoreEvent(CoreEventType.MESSAGE_DELTA, {"delta": event.delta}))
            elif event.event_type == StreamEventType.THINKING_DELTA:
                self.emit_event(CoreEvent(CoreEventType.THINKING_DELTA, {"delta": event.delta}))
            elif event.event_type == StreamEventType.MESSAGE_STOP and event.usage:
                turn.add_usage(event.usage)
            elif event.event_type == StreamEventType.ERROR:
                self.emit_event(CoreEvent(CoreEventType.ERROR, {"error": event.error}))
                break
        recovered = "".join(text_parts).strip()
        if recovered and not self._contains_dsml(recovered):
            return recovered
        return self._fallback_summary_from_thinking()

    def _fallback_summary_from_thinking(self) -> str:
        candidates = [
            str(event.payload.get("delta") or "")
            for event in self.core_events
            if event.event_type == CoreEventType.THINKING_DELTA
        ]
        combined = "".join(candidates).strip()
        if combined:
            marker = combined.rfind("让我")
            if marker > 40:
                combined = combined[:marker].strip()
            if combined:
                return combined
        return "Agent 已完成查询，但最终回答格式异常，请重新提问。"

    @staticmethod
    def _forced_final_summary_prompt() -> str:
        return (
            "这是本任务允许的最后一次模型调用。禁止调用任何工具，也不要提出新的检索计划。"
            "请仅基于当前会话中已经获得的工具结果和上下文，立即输出完整的最终回答。"
            "若证据存在不足、冲突或时效限制，必须在最终回答中明确说明，而不是继续查询。"
        )

    @staticmethod
    def _contains_dsml(text: str) -> bool:
        normalized = text or ""
        return "DSML" in normalized or "<｜｜" in normalized or "tool_calls>" in normalized

    def execute_tool_with_lock(self, plan: ToolExecutionPlan, turn: Turn) -> ToolExecutionOutcome:
        started = perf_counter()
        result = self.tool_registry.execute_full_with_context(
            plan.tool_name,
            plan.input,
            ToolContext(
                workspace_root=str(self.session.working_set.get("workspace_root", "")),
                session_id=str(self.session.settings.get("session_id", "")),
                turn_id=turn.id,
            ),
        )
        duration_ms = int((perf_counter() - started) * 1000)
        outcome = ToolExecutionOutcome(
            tool_use_id=plan.tool_use_id,
            result=result,
            duration_ms=duration_ms,
            is_error=result.is_error,
        )
        event_type = CoreEventType.TOOL_CALL_FAILED if result.is_error else CoreEventType.TOOL_CALL_COMPLETE
        self.emit_event(
            CoreEvent(
                event_type,
                {
                    "tool_use_id": outcome.tool_use_id,
                    "tool_name": plan.tool_name,
                    "arguments": plan.input,
                    "result": to_plain_dict(outcome.result),
                    "duration_ms": duration_ms,
                },
            )
        )
        return outcome

    def emit_event(self, event: CoreEvent) -> None:
        self.core_events.append(event)
        listener = self.event_listener
        if listener is None:
            return
        try:
            listener(event)
        except (BrokenPipeError, ConnectionError, OSError):
            # The Agent turn may still finish and be persisted after the browser disconnects.
            self.event_listener = None

    def add_session_message(
        self,
        role: str,
        content: str,
        turn_metadata: Optional[TurnMetadata] = None,
    ) -> None:
        if role == "user":
            self.session.messages.append_user_message(content, turn_metadata=turn_metadata)
        elif role == "assistant":
            self.session.messages.append_assistant_message(text=content)
        elif role == "tool":
            raise ValueError("Use MessageHistory.append_tool_result for tool messages")
        else:
            raise ValueError(f"Unsupported session message role: {role}")


def build_demo_engine(max_steps: int = 20) -> Engine:
    return Engine(
        session=Session(
            working_set={"workspace_root": "demo-workspace"},
            settings={
                "model": "demo-chat-model",
                "session_id": "session_demo",
                "current_date": "2026-07-08",
            },
        ),
        llm_client=ScriptedLlmClient(),
        tool_registry=build_demo_tool_registry(),
        stop_controller=MaxStepsStopController(max_steps=max_steps),
    )


def run_engine_loop_demo(max_steps: int = 20) -> Dict[str, Any]:
    engine = build_demo_engine(max_steps=max_steps)
    events = engine.handle_send_message("请调用 echo 工具，然后总结结果。")
    request_after_run = (
        MessageRequestBuilder(engine.session, engine.tool_registry, engine.adapter)
        .tool_choice("auto")
        .stream_true()
        .build()
    )
    return {
        "core_events": [to_plain_dict(event) for event in events],
        "session_messages": engine.adapter.message_history_to_chat_messages(engine.session.messages),
        "final_chat_completions_request": request_after_run,
    }
