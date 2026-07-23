"""读取自媒体选题 skill 的动态加载器。

正文由 skill_loader 工具按需加载，不固定放入 agent 的 system prompt。

项目依赖：由上层 `skill_loader_tool.py` 调用，并读取 research_outputs 下的 Skill 文件。
外部依赖：仅使用 Python 标准库 Path 和正则表达式。
"""

from __future__ import annotations

import re
from pathlib import Path


SKILL_FILE = (
    Path(__file__).resolve().parents[2]
    / "research_outputs"
    / "创作者画像与选题指标落地方案"
    / "skills"
    / "creator-topic-recommendation"
    / "SKILL.md"
)


def _remove_frontmatter(text: str) -> str:
    """skill 的 YAML 元数据由工具发现机制使用，不重复塞入模型正文。"""

    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL).strip()


def load_creator_topic_skill() -> str:
    """按 UTF-8 读取 skill 正文；文件缺失时返回可解释的最小提示。"""

    try:
        return _remove_frontmatter(SKILL_FILE.read_text(encoding="utf-8"))
    except OSError as exc:
        return "创作者选题 skill 当前不可用，错误：{}".format(exc)


def prompt_part(description: str, content: str) -> str:
    """生成统一的 system prompt 分段协议。"""

    return "###（{}）####\n{}\n########".format(description, content.strip())
