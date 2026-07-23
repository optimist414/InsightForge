# Profile Output Reference

## Minimal valid model output

```json
{
  "creator_identity_tags": ["科技区", "解释型"],
  "content_positioning": {
    "one_sentence": "用通俗语言解释人工智能产品和行业变化",
    "value_proposition": "帮助非专业观众理解技术影响",
    "style": ["解释型", "观点型"]
  },
  "content_format": {
    "primary": "long_video",
    "formats": ["长视频"],
    "duration_range_seconds": {"min": 480, "max": 1500, "median": 820}
  },
  "creator_size": {"level": "unknown", "follower_count": null, "median_views": 82000},
  "domain_profile": {
    "primary": "人工智能",
    "secondary": ["科技数码"],
    "main_tags": ["大模型", "AI产品"],
    "avoid_tags": ["娱乐八卦"]
  },
  "audience_profile": {
    "description": "关注 AI 但不一定具备技术背景的观众",
    "needs": ["理解概念", "判断影响"],
    "knowledge_level": "入门到中级"
  },
  "evidence_summary": {
    "sample_count": 20,
    "supported_claims": ["主要作品围绕大模型"],
    "unknowns": ["粉丝数未提供"]
  },
  "confidence": "medium"
}
```

`creator_profile_prompt.py` 会把该输出转换成下游统一对象。`llm_output` 只用于测试或重放，正常生产调用由工具内部发起独立 Chat Completions 请求。
