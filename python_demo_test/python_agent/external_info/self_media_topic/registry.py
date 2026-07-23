"""自媒体选题工具注册器。

作用：读取当前 `schemas.py` 和 `handlers.py`，把保留的业务工具注册到统一 ToolRegistry。
项目依赖：`schemas.py` 提供协议，`handlers.py` 提供执行函数，`bilibili_creator.py` 作为画像工具的内部采集适配器，`view.tooling` 提供注册器。
外部依赖：不直接访问外部服务。
"""

from __future__ import annotations

from ...view.tooling import BuiltinTool, ToolRegistry
from .handlers import HANDLER_BY_TOOL_NAME, wrap_handler
from .schemas import TOOL_DEFINITIONS


SELF_MEDIA_TOPIC_TOOL_NAMES = [
    *[definition.name for definition in TOOL_DEFINITIONS],
]


def register_self_media_topic_tools(registry: ToolRegistry) -> ToolRegistry:
    """把自媒体选题业务工具注册到统一 ToolRegistry。"""

    for definition in TOOL_DEFINITIONS:
        registry.register(
            BuiltinTool(
                tool_name=definition.name,
                tool_description=definition.description,
                schema=definition.input_schema,
                handler=wrap_handler(HANDLER_BY_TOOL_NAME[definition.name]),
            )
        )
    return registry


def build_self_media_topic_tool_registry() -> ToolRegistry:
    """构造只包含自媒体选题工具的 registry，便于测试和独立使用。"""

    return register_self_media_topic_tools(ToolRegistry())
