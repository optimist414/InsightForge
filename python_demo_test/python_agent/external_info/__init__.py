"""external_info 工具聚合包。

作用：集中导出并注册联网、Java SQL、动态 Skill 和自媒体选题工具。
项目依赖：本包下的 `web_tools`、`java_sql_proxy`、`skill_loader_tool` 和 `self_media_topic`。
外部依赖：由各子模块分别声明，聚合层本身不直接访问外部服务。
"""

from .self_media_topic import (
    build_self_media_topic_tool_registry,
    normalize_data_sources,
    normalize_source_bundle,
    register_self_media_topic_tools,
)
from .web_tools import build_web_tool_registry, register_web_tools
from .java_sql_proxy import (
    build_java_sql_proxy_tool_registry,
    query_java_sql_proxy,
    register_java_sql_proxy_tools,
)
from .skill_loader_tool import register_skill_loader_tool
from ..view.tooling import ToolRegistry


def build_external_tool_registry() -> ToolRegistry:
    """聚合 external_info 下所有业务工具。"""

    registry = ToolRegistry()
    register_web_tools(registry)
    register_java_sql_proxy_tools(registry)
    register_self_media_topic_tools(registry)
    register_skill_loader_tool(registry)
    return registry

__all__ = [
    "build_external_tool_registry",
    "build_web_tool_registry",
    "build_java_sql_proxy_tool_registry",
    "build_self_media_topic_tool_registry",
    "normalize_data_sources",
    "normalize_source_bundle",
    "register_self_media_topic_tools",
    "register_web_tools",
    "register_java_sql_proxy_tools",
    "query_java_sql_proxy",
    "register_skill_loader_tool",
]
