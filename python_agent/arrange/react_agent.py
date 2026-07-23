from __future__ import annotations

from .multi_turn_agent import (
    AgentRunResult,
    AgentTurnResult,
    MultiTurnAgent,
    ReactAgentMvp,
    build_demo_multi_turn_agent,
    run_arrange_n_turn_demo,
)

__all__ = [
    "AgentRunResult",
    "AgentTurnResult",
    "MultiTurnAgent",
    "ReactAgentMvp",
    "build_demo_multi_turn_agent",
    "run_arrange_n_turn_demo",
]

'''
import json
import re
from pathlib import Path
from typing import Any, List, Optional

from ..external_info.tool_executor import ToolExecutionLayer
from ..external_info.tool_registry import ToolRegistry, default_tool_registry
from ..memory.store import InMemoryMemoryStore
from ..tools.protocol import AgentOutputParseError, AgentStepOutputParser
from ..view.loop_view import LoopViewUpdater
from ..view.models import (
    Action,
    ActionType,
    AgentInput,
    AgentOutput,
    AgentStepOutput,
    ExecutionPhase,
    ExternalReference,
    FinalResponse,
    Goal,
    LoopContextSummary,
    LoopEvent,
    LoopEventType,
    LoopOutput,
    LoopWorkingMemory,
    OutputFormat,
    to_plain_dict,
)


class ReactAgentMvp:
    """Executable MVP for the proposed AgentInput/ReAct data flow."""

    def __init__(
        self,
        memory_store: Optional[InMemoryMemoryStore] = None,
        tool_registry: Optional[ToolRegistry] = None,
        planner: Optional[Any] = None,
        max_loops: int = 4,
    ) -> None:
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.tool_registry = tool_registry or default_tool_registry()
        self.planner = planner
        self.output_parser = AgentStepOutputParser()
        self.tool_executor = ToolExecutionLayer(self.tool_registry)
        self.loop_view_updater = LoopViewUpdater()
        self.max_loops = max_loops

    def build_input(self, user_input: str, external_refs: Optional[List[ExternalReference]] = None) -> AgentInput:
        goal = Goal(
            user_input=user_input,
            task_description=self._describe_task(user_input),
            constraints=[
                "Use structured AgentInput as the prompt-facing input view.",
                "Record every loop action, observation, error, and state update.",
                "Return a final response when no further action is needed.",
            ],
        )
        agent_input = AgentInput(
            goal=goal,
            memories=self.memory_store.select_for_agent_input(),
            external_refs=external_refs or [],
        )
        self._append_event(
            agent_input,
            LoopEventType.EXECUTION_STATE,
            "AgentInput assembled from goal, memories, and external references. Loop context starts empty.",
            0,
            {
                "goal": goal.task_description,
                "memory_count": len(agent_input.memories),
                "loop_context_is_empty": True,
            },
        )
        return agent_input

    def run(self, user_input: str, output_path: Optional[Path] = None) -> AgentOutput:
        agent_input = self.build_input(user_input)
        loop_outputs: List[LoopOutput] = []
        final_response: Optional[FinalResponse] = None

        for loop_index in range(1, self.max_loops + 1):
            context_summary = self._read_loop_context_summary(agent_input, loop_index)
            self._append_event(
                agent_input,
                LoopEventType.LOOP_CONTEXT_READ,
                context_summary.summary_text,
                loop_index,
                to_plain_dict(context_summary),
            )
            self._update_state(
                agent_input,
                ExecutionPhase.PLANNING,
                loop_index,
                f"Planning loop {loop_index}.",
            )
            raw_agent_output = self._generate_agent_step_output(
                agent_input,
                loop_outputs,
                loop_index,
                context_summary,
            )
            try:
                step_output = self.output_parser.parse(raw_agent_output)
            except AgentOutputParseError as exc:
                parse_error = exc.to_error_feedback(loop_index)
                self.loop_view_updater.record_error(agent_input, parse_error, loop_index)
                loop_outputs.append(
                    LoopOutput(
                        reasoning_text="Agent 输出协议解析失败。",
                        action=Action(action_type=ActionType.NO_OP, reason="Parser failure."),
                        loop_index=loop_index,
                        agent_step_output=AgentStepOutput(
                            thought="Agent 输出协议解析失败。",
                            action=Action(action_type=ActionType.NO_OP, reason="Parser failure."),
                            raw_text=raw_agent_output,
                            metadata={"parse_error": str(exc)},
                        ),
                        context_summary=context_summary,
                        error=parse_error,
                    )
                )
                continue

            reasoning_text = step_output.thought
            action = step_output.action
            self._append_event(
                agent_input,
                LoopEventType.REASONING_SUMMARY,
                reasoning_text,
                loop_index,
            )
            self._append_event(
                agent_input,
                LoopEventType.ACTION,
                f"Prepared action: {action.action_type.value}",
                loop_index,
                to_plain_dict(action),
            )

            if action.action_type == ActionType.FINAL_RESPONSE:
                final_response = FinalResponse(
                    format=OutputFormat.MARKDOWN,
                    content=action.arguments.get("content", ""),
                )
                self._append_event(
                    agent_input,
                    LoopEventType.FINAL_RESPONSE,
                    final_response.content,
                    loop_index,
                    to_plain_dict(final_response),
                )
                loop_outputs.append(
                    LoopOutput(
                        reasoning_text=reasoning_text,
                        action=action,
                        loop_index=loop_index,
                        agent_step_output=step_output,
                        context_summary=context_summary,
                    )
                )
                break

            loop_output = self._execute_action(
                agent_input,
                reasoning_text,
                action,
                loop_index,
                context_summary,
                step_output,
            )
            loop_outputs.append(loop_output)
            if loop_output.error and not loop_output.error.recoverable:
                final_response = FinalResponse(
                    format=OutputFormat.MARKDOWN,
                    content=f"任务失败：{loop_output.error.message}",
                )
                break

        if final_response is None:
            final_response = self._summarize(agent_input, loop_outputs)

        self._update_state(
            agent_input,
            ExecutionPhase.COMPLETED,
            len(loop_outputs),
            "Agent run completed.",
            percent=100.0,
        )
        agent_output = AgentOutput(
            loop_outputs=loop_outputs,
            final_response=final_response,
            metadata={
                "agent_input": to_plain_dict(agent_input),
                "tool_catalog": self.tool_registry.describe_detailed(),
                "planner": self.planner.__class__.__name__ if self.planner else "RuleBasedPlanner",
            },
        )
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(to_plain_dict(agent_output), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return agent_output

    def _execute_action(
        self,
        agent_input: AgentInput,
        reasoning_text: str,
        action: Action,
        loop_index: int,
        context_summary: LoopContextSummary,
        step_output: AgentStepOutput,
    ) -> LoopOutput:
        if action.action_type == ActionType.TOOL_CALL:
            self._update_state(
                agent_input,
                ExecutionPhase.TOOL_CALLING,
                loop_index,
                f"Calling tool: {action.tool_name}.",
            )
        elif action.action_type == ActionType.COMMAND_GENERATION:
            self._update_state(
                agent_input,
                ExecutionPhase.COMMAND_GENERATING,
                loop_index,
                "Generating command text.",
            )

        result = self.tool_executor.execute(action, loop_index)
        if result.observation:
            self.loop_view_updater.record_observation(agent_input, result.observation, loop_index)
        if result.external_reference:
            self.loop_view_updater.record_external_reference(agent_input, result.external_reference, loop_index)
        if result.error:
            self.loop_view_updater.record_error(agent_input, result.error, loop_index)
        if result.observation and result.observation.success:
            self._write_loop_memory(
                agent_input,
                loop_index,
                f"Tool {result.observation.tool_name} output",
                str(result.observation.output),
            )

        return LoopOutput(
            reasoning_text=reasoning_text,
            action=action,
            loop_index=loop_index,
            agent_step_output=step_output,
            context_summary=context_summary,
            observation=result.observation,
            error=result.error,
        )

    def _generate_agent_step_output(
        self,
        agent_input: AgentInput,
        loop_outputs: List[LoopOutput],
        loop_index: int,
        context_summary: LoopContextSummary,
    ) -> str:
        if self.planner:
            return self.planner.generate_step_output(
                agent_input=agent_input,
                loop_outputs=loop_outputs,
                loop_index=loop_index,
                context_summary=context_summary,
                tool_catalog=self.tool_registry.describe_detailed(),
            )
        thought, action = self._choose_next_action(
            agent_input,
            loop_outputs,
            loop_index,
            context_summary,
        )
        return json.dumps(
            {
                "thought": thought,
                "action": to_plain_dict(action),
            },
            ensure_ascii=False,
        )

    def _choose_next_action(
        self,
        agent_input: AgentInput,
        loop_outputs: List[LoopOutput],
        loop_index: int,
        context_summary: LoopContextSummary,
    ) -> tuple[str, Action]:
        user_input = agent_input.goal.user_input
        if loop_outputs:
            called_tools = {item.action.tool_name for item in loop_outputs if item.action.tool_name}
            if self._looks_like_complex_mcp_task(user_input):
                if "mcp_requirement_analyzer" not in called_tools:
                    return (
                        "复杂任务需要先拆解需求，调用 MCP 需求分析工具。",
                        Action(
                            action_type=ActionType.TOOL_CALL,
                            tool_name="mcp_requirement_analyzer",
                            arguments={"task": user_input},
                            reason="Complex task requires requirement analysis.",
                        ),
                    )
                if "mcp_hot_news_schema" not in called_tools:
                    return (
                        "已有需求拆解，继续调用热点新闻库表结构 MCP，补齐表字段和关联规则。",
                        Action(
                            action_type=ActionType.TOOL_CALL,
                            tool_name="mcp_hot_news_schema",
                            arguments={"focus": user_input},
                            reason="Need schema and join rules.",
                        ),
                    )
                if "mcp_platform_policy" not in called_tools:
                    return (
                        "已有库表信息，继续调用平台规则 MCP，补齐平台代码和标签关键词映射。",
                        Action(
                            action_type=ActionType.TOOL_CALL,
                            tool_name="mcp_platform_policy",
                            arguments={"platforms": ["知乎", "微博"]},
                            reason="Need platform codes and tag rules.",
                        ),
                    )
                if "mcp_query_plan_quality_check" not in called_tools:
                    return (
                        "已有需求、库表和平台规则，调用查询方案质检 MCP 检查遗漏。",
                        Action(
                            action_type=ActionType.TOOL_CALL,
                            tool_name="mcp_query_plan_quality_check",
                            arguments={"plan": self._render_final_content(agent_input, loop_outputs)},
                            reason="Quality check before final response.",
                        ),
                    )
            return (
                f"读取到历史 loop 摘要：{context_summary.summary_text}。已有工具反馈，进入总结阶段，生成最终回复。",
                Action(
                    action_type=ActionType.FINAL_RESPONSE,
                    arguments={"content": self._render_final_content(agent_input, loop_outputs)},
                    reason="Task has enough observations for a final response.",
                ),
            )

        expression = self._extract_expression(user_input)
        if expression:
            return (
                "用户输入包含可计算表达式，选择 calculator 工具获取结构化观察结果。",
                Action(
                    action_type=ActionType.TOOL_CALL,
                    tool_name="calculator",
                    arguments={"expression": expression},
                    reason="Numeric expression detected.",
                ),
            )

        if self._looks_like_complex_mcp_task(user_input):
            return (
                "复杂任务需要先拆解需求，调用 MCP 需求分析工具。",
                Action(
                    action_type=ActionType.TOOL_CALL,
                    tool_name="mcp_requirement_analyzer",
                    arguments={"task": user_input},
                    reason="Complex task requires requirement analysis.",
                ),
            )

        if any(keyword.lower() in user_input.lower() for keyword in ["mcp", "skill", "外部参考", "外部工具"]):
            return (
                "用户意图需要外部参考信息，选择 demo_mcp_reference 工具，并将返回内容附到 loop 视图。",
                Action(
                    action_type=ActionType.TOOL_CALL,
                    tool_name="demo_mcp_reference",
                    arguments={"topic": user_input},
                    reason="External reference keyword detected.",
                ),
            )

        if any(keyword in user_input for keyword in ["命令", "指令", "shell", "执行"]):
            return (
                "用户意图更接近指令生成，生成命令文本但不执行。",
                Action(
                    action_type=ActionType.COMMAND_GENERATION,
                    arguments={"task": user_input},
                    reason="Command generation keyword detected.",
                ),
            )

        return (
            "当前任务不需要外部工具，使用 echo 工具把输入转成观察结果，验证 loop 数据链路。",
            Action(
                action_type=ActionType.TOOL_CALL,
                tool_name="echo",
                arguments={"text": user_input},
                reason="Default MVP observation path.",
            ),
        )

    def _read_loop_context_summary(
        self,
        agent_input: AgentInput,
        current_loop_index: int,
    ) -> LoopContextSummary:
        prior_events = [
            event
            for event in agent_input.loop_context.events
            if 0 < event.loop_index < current_loop_index
        ]
        prior_observations = [
            observation
            for observation in agent_input.loop_context.tool_observations
            if observation.metadata.get("loop_index", 0) < current_loop_index
        ]
        prior_errors = [
            error
            for error in agent_input.loop_context.errors
            if error.metadata.get("loop_index", 0) < current_loop_index
        ]
        prior_external_refs = [
            reference
            for reference in agent_input.loop_context.external_refs
            if reference.metadata.get("loop_index", 0) < current_loop_index
        ]
        prior_memories = [
            memory
            for memory in agent_input.loop_context.loop_memories
            if memory.loop_index < current_loop_index
        ]
        prior_loop_indexes = sorted(
            {event.loop_index for event in prior_events}
            | {memory.loop_index for memory in prior_memories}
            | {observation.metadata.get("loop_index", 0) for observation in prior_observations}
            | {reference.metadata.get("loop_index", 0) for reference in prior_external_refs}
        )

        if not prior_events and not prior_observations and not prior_errors and not prior_external_refs and not prior_memories:
            return LoopContextSummary(current_loop_index=current_loop_index)

        latest_state = agent_input.loop_context.execution_state.progress_summary
        memory_summaries = [
            f"Loop {memory.loop_index}: {memory.title}={memory.content}"
            for memory in prior_memories[-5:]
        ]
        external_reference_summaries = [
            f"Loop {reference.metadata.get('loop_index')}: {reference.title} - {reference.content}"
            for reference in prior_external_refs[-5:]
        ]
        event_summaries = [
            f"Loop {event.loop_index} {event.event_type.value}: {event.content}"
            for event in prior_events[-8:]
        ]
        summary_parts = [
            f"prior_loops={len([index for index in prior_loop_indexes if index > 0])}",
            f"observations={len(prior_observations)}",
            f"errors={len(prior_errors)}",
            f"external_refs={len(prior_external_refs)}",
            f"loop_memories={len(prior_memories)}",
        ]
        if external_reference_summaries:
            summary_parts.append(f"latest_external_ref={external_reference_summaries[-1]}")
        if memory_summaries:
            summary_parts.append(f"latest_memory={memory_summaries[-1]}")

        return LoopContextSummary(
            current_loop_index=current_loop_index,
            prior_loop_count=len([index for index in prior_loop_indexes if index > 0]),
            summary_text="; ".join(summary_parts),
            is_empty=False,
            tool_observation_count=len(prior_observations),
            error_count=len(prior_errors),
            external_reference_count=len(prior_external_refs),
            loop_memory_count=len(prior_memories),
            latest_state=latest_state,
            loop_memory_summaries=memory_summaries,
            external_reference_summaries=external_reference_summaries,
            event_summaries=event_summaries,
        )

    def _summarize(self, agent_input: AgentInput, loop_outputs: List[LoopOutput]) -> FinalResponse:
        return FinalResponse(
            format=OutputFormat.MARKDOWN,
            content=self._render_final_content(agent_input, loop_outputs),
        )

    def _render_final_content(self, agent_input: AgentInput, loop_outputs: List[LoopOutput]) -> str:
        lines = [
            f"任务：{agent_input.goal.task_description}",
            "",
            "Loop 结果：",
        ]
        for item in loop_outputs:
            lines.append(f"- Loop {item.loop_index}: {item.reasoning_text}")
            if item.observation:
                lines.append(f"  - 工具：{item.observation.tool_name}")
                lines.append(f"  - 成功：{item.observation.success}")
                lines.append(f"  - 输出：{item.observation.output}")
            if item.error:
                lines.append(f"  - 报错：{item.error.message}")
        return "\n".join(lines)

    def _describe_task(self, user_input: str) -> str:
        return f"处理用户输入并通过 ReAct loop 生成可追踪输出：{user_input}"

    def _extract_expression(self, text: str) -> str:
        for match in re.finditer(r"[-+*/().\d\s]+", text):
            expression = match.group(0).strip()
            if any(ch.isdigit() for ch in expression) and any(op in expression for op in "+-*/"):
                return expression
        return ""

    def _looks_like_complex_mcp_task(self, text: str) -> bool:
        lowered = text.lower()
        markers = ["复杂", "方案", "设计", "字段", "过滤", "排序", "展示格式", "表", "mcp"]
        hot_news_markers = ["热点", "新闻", "知乎", "微博", "ai", "AI"]
        return sum(marker in text or marker in lowered for marker in markers) >= 2 and any(
            marker in text or marker in lowered for marker in hot_news_markers
        )

    def _update_state(
        self,
        agent_input: AgentInput,
        status: ExecutionPhase,
        step: int,
        summary: str,
        percent: Optional[float] = None,
    ) -> None:
        self.loop_view_updater.update_state(
            agent_input,
            status,
            step,
            summary,
            percent,
        )

    def _write_loop_memory(self, agent_input: AgentInput, loop_index: int, title: str, content: str) -> None:
        self.loop_view_updater.write_loop_memory(agent_input, loop_index, title, content)

    def _append_event(
        self,
        agent_input: AgentInput,
        event_type: LoopEventType,
        content: str,
        loop_index: int,
        payload: Optional[dict] = None,
    ) -> None:
        self.loop_view_updater.record_event(agent_input, event_type, content, loop_index, payload)
'''
