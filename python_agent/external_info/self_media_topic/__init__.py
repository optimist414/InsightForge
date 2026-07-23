"""自媒体选题业务包。

作用：导出数据归一化能力和当前保留的工具注册入口。
项目依赖：`domain.py`、`registry.py` 及其下游 schema/handler 模块。
外部依赖：由具体业务模块按需访问 Bocha、Java SQL 或 LLM 服务。
"""

from .domain import (
    normalize_data_sources,
    normalize_source_bundle,
    fetch_web_search_sources,
)
from .registry import (
    SELF_MEDIA_TOPIC_TOOL_NAMES,
    build_self_media_topic_tool_registry,
    register_self_media_topic_tools,
)

__all__ = [
    "SELF_MEDIA_TOPIC_TOOL_NAMES",
    "build_self_media_topic_tool_registry",
    "fetch_web_search_sources",
    "normalize_data_sources",
    "normalize_source_bundle",
    "register_self_media_topic_tools",
]
