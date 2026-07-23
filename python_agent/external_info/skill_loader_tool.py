"""动态 skill 加载工具。

skill 正文不放入 system prompt。模型先通过本工具查看目录，再按需加载；
Engine 会把加载正文作为 assistant 消息追加到当前会话。

项目依赖：`self_media_topic.skill_loader` 负责读取 Skill，`view.tooling` 负责工具协议。
外部依赖：仅使用 Python 标准库。
"""

from __future__ import annotations

import json
from typing import Any, Dict

from .self_media_topic.skill_loader import load_creator_topic_skill
from ..view.tooling import BuiltinTool, ToolContext, ToolRegistry, ToolResult


SKILL_LOADER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "load"],
            "description": "list 查看可用 skill；load 加载指定 skill，默认 list。",
        },
        "skill_name": {
            "type": "string",
            "description": "action=load 时填写 skill 名称，例如 creator-topic-recommendation。",
        },
    },
    "additionalProperties": False,
}

SKILL_CATALOG = [
    {
        "name": "creator-topic-recommendation",
        "description": "根据创作者历史作品、项目热点和联网证据生成创作选题建议。",
        "when_to_load": "用户要求创作者画像、内容定位、长短视频判断、选题推荐或创作角度时加载。",
        "how_to_load": "调用 skill_loader，传入 {\"action\":\"load\",\"skill_name\":\"creator-topic-recommendation\"}。",
    }
]

SKILL_LOADER_DESCRIPTION = (
    "动态 skill 加载器。当前可用 skill：creator-topic-recommendation。"
    "该 skill 在用户要求创作者画像、内容定位、长短视频判断、选题推荐或创作角度时加载；"
    "加载方式：先调用 {action:list} 查看目录，确认后调用 "
    "{action:load,skill_name:creator-topic-recommendation}。"
    "加载正文会在下一次模型请求中以 assistant 消息提供，不会写入 system prompt。"
)


def execute_skill_loader(input: Dict[str, Any], _ctx: ToolContext) -> ToolResult:
    action = str(input.get("action") or "list").strip().lower()
    if action == "list":
        payload = {"available_skills": SKILL_CATALOG, "usage": "action=load + skill_name 加载 skill。"}
        return ToolResult(content=json.dumps(payload, ensure_ascii=False, indent=2), metadata={"skill_action": "list"})
    if action != "load":
        return ToolResult(content="skill_loader action must be list or load", is_error=True)

    skill_name = str(input.get("skill_name") or "").strip()
    if skill_name != "creator-topic-recommendation":
        return ToolResult(
            content=json.dumps({"error": "unknown skill", "available_skills": SKILL_CATALOG}, ensure_ascii=False),
            is_error=True,
        )
    assistant_message = "[Loaded Skill: creator-topic-recommendation]\n" + load_creator_topic_skill()
    return ToolResult(
        content=json.dumps(
            {
                "status": "loaded",
                "skill_name": skill_name,
                "message": "skill 已加载，将作为 assistant 消息追加到下一次模型请求。",
            },
            ensure_ascii=False,
        ),
        metadata={
            "skill_action": "load",
            "skill_name": skill_name,
            "assistant_message": assistant_message,
        },
    )


def register_skill_loader_tool(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        BuiltinTool(
            tool_name="skill_loader",
            tool_description=SKILL_LOADER_DESCRIPTION,
            schema=SKILL_LOADER_SCHEMA,
            handler=execute_skill_loader,
        )
    )
    return registry
