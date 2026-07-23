from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..engine.core import (
    CoreEvent,
    CoreEventType,
    Engine,
    LlmClient,
    MaxStepsStopController,
    MessageRequestBuilder,
    ScriptedLlmClient,
    Session,
    StopController,
)
from ..view.prompt import to_plain_dict
from ..view.tooling import ToolRegistry, build_demo_tool_registry


TurnStopControllerFactory = Callable[[int, str], StopController]


@dataclass
class AgentTurnResult:
    """单个 turn 的编排结果。"""

    turn_index: int
    user_text: str
    stop_controller: str
    stop_reason: str
    event_count: int
    message_count_after_turn: int
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentRunResult:
    """多 turn agent 的整体运行结果。"""

    requested_turn_count: int
    executed_turn_count: int
    max_turns: int
    turns: List[AgentTurnResult]
    session_messages: List[Dict[str, Any]]
    final_chat_completions_request: Dict[str, Any]


class MultiTurnAgent:
    """Arrange layer for a multi-turn agent over one shared session.

    `Engine` is responsible for one user turn and its internal tool loop.
    `MultiTurnAgent` owns the conversation session: every `send_message()`
    appends one more user turn to the same session, then creates a fresh
    per-turn stop controller for that turn.
    """

    def __init__(
        self,
        session: Optional[Session] = None,
        llm_client: Optional[LlmClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        turn_stop_controller_factory: Optional[TurnStopControllerFactory] = None,
        max_turns: int = 3,
    ) -> None:
        self.session = session or self._default_session()
        self.llm_client = llm_client or ScriptedLlmClient()
        self.tool_registry = tool_registry or build_demo_tool_registry()
        self.turn_stop_controller_factory = turn_stop_controller_factory or (
            lambda _turn_index, _user_text: MaxStepsStopController(max_steps=20)
        )
        self.max_turns = max_turns
        self.turn_index = 0
        self.turn_results: List[AgentTurnResult] = []

    def send_message(
        self,
        user_text: str,
        event_listener: Optional[Callable[[CoreEvent], None]] = None,
    ) -> AgentTurnResult:
        """Append one user message to the shared session and run one turn."""

        if self.turn_index >= self.max_turns:
            raise RuntimeError(f"Max turns reached: {self.max_turns}")
        self.turn_index += 1
        result = self.run_turn(
            user_text=user_text,
            turn_index=self.turn_index,
            event_listener=event_listener,
        )
        self.turn_results.append(result)
        return result

    def run_turn(
        self,
        user_text: str,
        turn_index: int,
        event_listener: Optional[Callable[[CoreEvent], None]] = None,
    ) -> AgentTurnResult:
        stop_controller = self.turn_stop_controller_factory(turn_index, user_text)
        engine = Engine(
            session=self.session,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            stop_controller=stop_controller,
            event_listener=event_listener,
        )
        events = engine.handle_send_message(user_text)
        return AgentTurnResult(
            turn_index=turn_index,
            user_text=user_text,
            stop_controller=stop_controller.__class__.__name__,
            stop_reason=self._infer_stop_reason(events),
            event_count=len(events),
            message_count_after_turn=len(self.session.messages.messages),
            events=[to_plain_dict(event) for event in events],
        )

    def run_turns(
        self,
        user_inputs: Sequence[str] | Iterable[str],
        max_turns: Optional[int] = None,
    ) -> AgentRunResult:
        limit = max_turns if max_turns is not None else self.max_turns
        selected_inputs = list(user_inputs)[:limit]
        turns = [self.send_message(user_text) for user_text in selected_inputs]
        final_request = (
            MessageRequestBuilder(self.session, self.tool_registry)
            .tool_choice("auto")
            .stream_true()
            .build()
        )
        adapter = MessageRequestBuilder(self.session, self.tool_registry).adapter
        return AgentRunResult(
            requested_turn_count=len(selected_inputs),
            executed_turn_count=len(turns),
            max_turns=limit,
            turns=turns,
            session_messages=adapter.message_history_to_chat_messages(self.session.messages),
            final_chat_completions_request=final_request,
        )

    def run(self, user_inputs: Sequence[str] | Iterable[str], max_turns: Optional[int] = None) -> AgentRunResult:
        return self.run_turns(user_inputs, max_turns=max_turns)

    @staticmethod
    def _infer_stop_reason(events: List[CoreEvent]) -> str:
        if any(event.event_type == CoreEventType.ERROR for event in events):
            return "error"
        if any(event.event_type == CoreEventType.MESSAGE_COMPLETE for event in events):
            return "assistant_message_complete"
        return "stop_controller"

    @staticmethod
    def _default_session() -> Session:
        return Session(
            working_set={"workspace_root": "demo-workspace"},
            settings={
                "model": "demo-chat-model",
                "session_id": "session_arrange_demo",
                "current_date": "2026-07-08",
            },
        )


def build_demo_multi_turn_agent(max_turns: int = 3, max_steps_per_turn: int = 20) -> MultiTurnAgent:
    return MultiTurnAgent(
        max_turns=max_turns,
        turn_stop_controller_factory=lambda _turn_index, _user_text: MaxStepsStopController(
            max_steps=max_steps_per_turn
        ),
    )


def run_arrange_n_turn_demo(turns: int = 3, max_steps_per_turn: int = 20) -> Dict[str, Any]:
    agent = build_demo_multi_turn_agent(max_turns=turns, max_steps_per_turn=max_steps_per_turn)
    user_inputs = [f"第{index}轮：请调用 echo 工具，然后总结结果。" for index in range(1, turns + 1)]
    result = agent.run(user_inputs, max_turns=turns)
    return to_plain_dict(result)


# Backward-compatible public name for older package-level imports.
ReactAgentMvp = MultiTurnAgent
