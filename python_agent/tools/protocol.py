from __future__ import annotations

import json
import re
from typing import Any, Dict

from ..view.models import Action, ActionType, AgentStepOutput, ErrorFeedback


class AgentStepOutputParser:
    """Parse the agent-facing Thought+Action protocol.

    Expected raw text:
    {
      "thought": "...",
      "action": {
        "action_type": "tool_call",
        "tool_name": "calculator",
        "arguments": {"expression": "1 + 2"},
        "reason": "..."
      }
    }
    """

    def parse(self, raw_text: str) -> AgentStepOutput:
        json_text = self._extract_json_object(raw_text)
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise AgentOutputParseError(
                message=f"Agent output is not valid JSON: {exc.msg}",
                raw_text=raw_text,
            ) from exc

        if not isinstance(payload, dict):
            raise AgentOutputParseError("Agent output must be a JSON object.", raw_text)

        thought = self._read_required_string(payload, "thought")
        action_payload = payload.get("action")
        if not isinstance(action_payload, dict):
            raise AgentOutputParseError("Agent output action must be an object.", raw_text)

        action = self._parse_action(action_payload, raw_text)
        return AgentStepOutput(thought=thought, action=action, raw_text=raw_text)

    def _extract_json_object(self, raw_text: str) -> str:
        text = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return text

    def _parse_action(self, payload: Dict[str, Any], raw_text: str) -> Action:
        action_type_text = self._read_required_string(payload, "action_type")
        try:
            action_type = ActionType(action_type_text)
        except ValueError as exc:
            raise AgentOutputParseError(f"Unsupported action_type: {action_type_text}", raw_text) from exc

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise AgentOutputParseError("Action arguments must be an object.", raw_text)

        tool_name = payload.get("tool_name", "")
        if tool_name is None:
            tool_name = ""
        if not isinstance(tool_name, str):
            raise AgentOutputParseError("Action tool_name must be a string.", raw_text)

        reason = payload.get("reason", "")
        if reason is None:
            reason = ""
        if not isinstance(reason, str):
            raise AgentOutputParseError("Action reason must be a string.", raw_text)

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise AgentOutputParseError("Action metadata must be an object.", raw_text)

        return Action(
            action_type=action_type,
            tool_name=tool_name,
            arguments=arguments,
            reason=reason,
            metadata=metadata,
        )

    def _read_required_string(self, payload: Dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AgentOutputParseError(f"Agent output field is required: {key}", str(payload))
        return value.strip()


class AgentOutputParseError(ValueError):
    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text

    def to_error_feedback(self, loop_index: int) -> ErrorFeedback:
        return ErrorFeedback(
            source="agent_output_parser",
            message=str(self),
            recoverable=True,
            suggested_fix="Return JSON with fields: thought and action.",
            metadata={"loop_index": loop_index, "raw_text": self.raw_text},
        )
