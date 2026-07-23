"""自媒体选题工具协议定义。

作用：声明当前 Agent 可见工具的名称、描述、输入 JSON Schema 和备注，不包含执行逻辑。
项目依赖：由 `registry.py` 读取并注册，执行函数位于 `handlers.py`。
外部依赖：仅使用 Python 标准库 dataclasses 和 typing。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolDefinition:
    """业务工具定义：只描述 agent 可见协议，不包含执行逻辑。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    note: str = ""


def object_schema(
    required: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
    any_of: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """构造宽松 object schema，允许业务字段继续扩展。"""

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": True,
    }
    if required:
        schema["required"] = required
    if any_of:
        schema["anyOf"] = any_of
    return schema


DATA_SOURCE_SCHEMA = object_schema(
    required=["source_type", "records"],
    properties={
        "source_id": {"type": "string", "description": "数据源稳定标识。"},
        "source_type": {"type": "string", "description": "database | search | export | api | manual 等。"},
        "source_name": {"type": "string", "description": "数据源展示名。"},
        "records": {"type": "array", "items": {"type": "object"}, "description": "原始记录列表。"},
    },
)

EVENT_CLUSTER_SCHEMA = {
    "type": "object",
    "description": "业务级热点事件簇，通常包含 event_id、representative_title、tags、items。",
    "additionalProperties": True,
}

CREATOR_PROFILE_SCHEMA = {
    "type": "object",
    "description": "经过 LLM JSON 输出协议解析后的创作者画像；下游指标只消费此结构。",
    "additionalProperties": True,
}


TOOL_DEFINITIONS = [
      ToolDefinition(
          name="creator_profile_builder",
          description="根据创作者标识和账号信息生成结构化创作者画像。",
          note="必须传入 creator_url，或同时传入 creator_id 和 platform；可选传入 explicit_profile 补充账号信息。",
          input_schema=object_schema(
              # 至少提供创作者标识或账号信息，禁止空对象调用。
              any_of=[
                  {"required": ["explicit_profile"]},
                  {"required": ["creator_id", "platform"]},
                  {"required": ["creator_url"]},
              ],
              properties={
                  "creator_id": {"type": "string", "description": "创作者平台内的用户 ID。"},
                  "platform": {"type": "string", "description": "创作者所属平台，例如 bilibili。"},
                  "creator_url": {"type": "string", "description": "创作者主页链接。"},
                  "explicit_profile": {"type": "object", "description": "用户已知的创作者账号补充信息。", "additionalProperties": True},
              },
        ),
    ),
    ToolDefinition(
        name="hot_news_fetcher",
        description="分两阶段从项目数据库读取创作者选题所需的热点候选和排名证据。",
        note=(
            "stage=candidates 时按 1 到 3 个 tag_ids 召回最多 50 条候选，只查询 hot_news 和 hot_news_tag；"
            "stage=ranking 时传入筛选后的 candidate_ids，最多 20 个，只查询 hot_news_record，最多返回 100 条。"
            "不要传入原始 SQL。"
        ),
        input_schema=object_schema(
            required=["stage"],
            properties={
                "stage": {
                    "type": "string",
                    "enum": ["candidates", "ranking"],
                    "default": "candidates",
                    "description": "查询阶段：先 candidates，再根据候选结果调用 ranking。",
                },
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "platforms": {"type": "array", "items": {"type": "string"}},
                "tag_ids": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 15},
                    "maxItems": 3,
                    "description": "候选阶段使用的标签 ID，来自 skill 固定标签目录。",
                },
                "candidate_ids": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "maxItems": 20,
                    "description": "排名阶段使用的候选 hot_news.id，通常从 candidates 结果中筛选 10 到 20 个。",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 50},
                "base_url": {"type": "string"},
            },
        ),
    ),
    ToolDefinition(
        name="proposal_topic_value_scorer",
        description="批量调用 LLM 评估热点与创作者画像的语义匹配度，再结合热度、上升、可创作性和风险折扣计算选题价值。",
        note=(
            "必须传入 LLM 解析后的 creator_profile。支持 event_cluster 单事件兼容输入，"
            "也支持 event_clusters 批量输入；批量输入只进行一次语义匹配调用。"
        ),
        input_schema=object_schema(
            any_of=[{"required": ["event_cluster", "creator_profile"]}, {"required": ["event_clusters", "creator_profile"]}],
            properties={
                "event_cluster": EVENT_CLUSTER_SCHEMA,
                "event_clusters": {"type": "array", "items": EVENT_CLUSTER_SCHEMA, "minItems": 1, "maxItems": 50},
                "creator_profile": CREATOR_PROFILE_SCHEMA,
                "search_records": {"type": "array", "items": {"type": "object"}},
                "candidate_angles": {"type": "array", "items": {"type": "object"}, "maxItems": 5},
            },
        ),
    ),
    ToolDefinition(
        name="content_angle_planner",
        description="根据事件和已解析创作者画像生成最多 5 个可执行内容角度，并说明适合的内容形式。",
        note="角度是创作计划，不是事实证据；事实仍需来源支持。",
        input_schema=object_schema(
            required=["event_cluster", "creator_profile"],
            properties={
                "event_cluster": EVENT_CLUSTER_SCHEMA,
                "creator_profile": CREATOR_PROFILE_SCHEMA,
                "count": {"type": "integer", "minimum": 1, "maximum": 5},
            },
        ),
    ),
    ToolDefinition(
        name="risk_evidence_checker",
        description="检查事件的敏感风险标记、来源数量和待补证据，供最终推荐前使用。",
        note="风险检查不能替代法律、医疗或投资专业判断。",
        input_schema=object_schema(
            required=["event_cluster"],
            properties={
                "event_cluster": EVENT_CLUSTER_SCHEMA,
                "search_records": {"type": "array", "items": {"type": "object"}},
                "score_result": {"type": "object", "additionalProperties": True},
            },
        ),
    ),
    ToolDefinition(
        name="creator_topic_recommender",
        description="对多个事件按 proposal_v1 排序，连接创作者画像、数据库热点和联网证据，返回推荐清单。",
        note="推荐前应先完成 creator_profile_builder 和数据源获取；每个事件最多处理 50 个。",
        input_schema=object_schema(
            required=["events", "creator_profile"],
            properties={
                "events": {"type": "array", "items": EVENT_CLUSTER_SCHEMA, "maxItems": 50},
                "creator_profile": CREATOR_PROFILE_SCHEMA,
                "search_records": {"type": "array", "items": {"type": "object"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
        ),
    ),
    ToolDefinition(
        name="web_search_fetcher",
        description="使用 Bocha Web Search 获取数据库热点的外部补充证据、趋势线索和权威来源，并返回 search_records。",
        note="联网搜索只补充数据库事件，不参与 source_bundle_normalizer 的标题聚类；搜索摘要不能直接当作已核实事实。",
        input_schema=object_schema(
            required=["queries"],
            properties={
                "queries": {
                    "type": "array",
                    "description": "搜索 query 列表。",
                    "items": {"type": "string"},
                },
                "count": {"type": "integer", "description": "每个 query 返回条数，默认 10，最大 20。"},
                "summary": {"type": "boolean", "description": "是否请求 Bocha 生成摘要，默认 true。"},
                "freshness": {"type": "string", "description": "可选时效过滤参数，透传给 Bocha。"},
            },
        ),
    ),
    ToolDefinition(
        name="source_bundle_normalizer",
        description=(
            "对数据库 hot_news 记录进行标题向量聚类，生成事件簇；联网搜索记录不参与聚类，"
            "仅作为事件的补充证据返回。默认使用本地 BGE 标题向量和余弦相似度。"
        ),
        note=(
            "输入应包含 hot_news_fetcher 返回的 database 数据；可同时传入 web_search_fetcher 返回的 search 数据，"
            "但它们只会输出到 supplementary_sources/search_records。cluster_threshold 默认 0.86，"
            "超过阈值的数据库标题通过两两比较合并为同一事件簇。"
        ),
        input_schema=object_schema(
            required=["data_sources"],
            properties={
                "data_sources": {
                    "type": "array",
                    "description": "多源输入，每项包含 source_type 和 records。",
                    "items": DATA_SOURCE_SCHEMA,
                },
                "cluster_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.86,
                    "description": "数据库标题向量余弦相似度阈值，超过该值的记录进入同一事件簇。",
                },
            },
        ),
    ),

]

TOOL_DEFINITION_BY_NAME = {definition.name: definition for definition in TOOL_DEFINITIONS}
