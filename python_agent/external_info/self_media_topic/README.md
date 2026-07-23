# 创作者选题工具包

该目录实现当前 `creator-topic-recommendation` Skill 使用的业务工具。

| 文件 | 职责 |
| --- | --- |
| `domain.py` | 数据源归一化与业务搜索结果适配。 |
| `schemas.py` | Agent 可见的工具名称、说明和输入 schema。 |
| `handlers.py` | 解析工具输入并统一包装 `ToolResult`。 |
| `proposal_metrics.py` | `proposal_v1` 选题评分、内容角度和证据风险判断。 |
| `creator_profile_prompt.py` | 创作者画像提示词、LLM 调用和结构化解析。 |
| `registry.py` | 将当前业务工具注册到统一 `ToolRegistry`。 |

## 当前流程

```text
读取创作者公开作品
-> creator_profile_builder
-> 按标签分阶段查询热点与排名
-> web_search / fetch_url 补充外部证据
-> source_bundle_normalizer
-> proposal_topic_value_scorer / creator_topic_recommender
-> content_angle_planner
-> risk_evidence_checker
```

当前决策算法统一使用 `proposal_v1`。工具目录不再暴露早期的事件信号、搜索需求、创作者匹配规则工具和一键 payload 聚合入口。
