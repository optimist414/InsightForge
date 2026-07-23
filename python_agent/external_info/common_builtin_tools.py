"""通用内置工具集合。

作用：提供受限算术计算和文本/文档检索等不依赖外部服务的基础工具。
项目依赖：`view.tooling` 的 BuiltinTool、ToolContext、ToolRegistry 和 ToolResult。
外部依赖：仅使用 Python 标准库，表达式计算通过 `ast` 白名单解析。
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from typing import Any, Dict, List, Tuple

from ..view.tooling import BuiltinTool, ToolCapabilities, ToolContext, ToolRegistry, ToolResult


CALCULATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["expression"],
    "properties": {
        "expression": {
            "type": "string",
            "description": "Arithmetic expression using numbers and + - * / // % ** parentheses.",
        }
    },
    "additionalProperties": False,
}

TEXT_SEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["query", "documents"],
    "properties": {
        "query": {"type": "string", "description": "Keyword or regular expression to search."},
        "documents": {
            "type": "array",
            "description": "Documents to search. Each item may contain id/title/text.",
            "items": {"type": "object", "additionalProperties": True},
        },
        "regex": {"type": "boolean", "description": "Treat query as regex. Defaults to false."},
        "case_sensitive": {"type": "boolean", "description": "Case-sensitive search. Defaults to false."},
        "max_matches": {"type": "integer", "description": "Maximum matches to return. Defaults to 5."},
    },
    "additionalProperties": True,
}

JSON_GET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["path"],
    "properties": {
        "json_text": {"type": "string", "description": "JSON document as text."},
        "data": {"description": "JSON document as an object or array."},
        "path": {"type": "string", "description": "Path like $.user.name, user.name, or $.items[0].title."},
    },
    "additionalProperties": True,
}

TABLE_SUMMARIZE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["rows"],
    "properties": {
        "rows": {"type": "array", "items": {"type": "object"}, "description": "Rows to summarize."},
        "fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to include."},
        "limit": {"type": "integer", "description": "Sample row limit. Defaults to 5."},
    },
    "additionalProperties": True,
}


def execute_calculate(input: Dict[str, Any], _ctx: ToolContext) -> ToolResult:
    try:
        expression = str(input.get("expression") or "").strip()
        if not expression:
            raise ValueError("expression is required")
        value = _safe_eval_expression(expression)
        payload = {"expression": expression, "result": value}
        return ToolResult(content=json.dumps(payload, ensure_ascii=False), metadata={"result": payload})
    except Exception as exc:  # noqa: BLE001 - tool boundary should return structured errors.
        return ToolResult(
            content=f"calculate failed: {exc}",
            is_error=True,
            metadata={"error_type": exc.__class__.__name__},
        )


def execute_text_search(input: Dict[str, Any], _ctx: ToolContext) -> ToolResult:
    try:
        query = str(input.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        documents = _coerce_documents(input.get("documents"))
        max_matches = max(1, min(int(input.get("max_matches") or 5), 50))
        regex = bool(input.get("regex", False))
        case_sensitive = bool(input.get("case_sensitive", False))
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(query if regex else re.escape(query), flags)
        matches: List[Dict[str, Any]] = []
        for doc in documents:
            text = str(doc.get("text") or "")
            for match in pattern.finditer(text):
                matches.append(
                    {
                        "document_id": str(doc.get("id") or ""),
                        "title": str(doc.get("title") or ""),
                        "start": match.start(),
                        "end": match.end(),
                        "snippet": _snippet(text, match.start(), match.end()),
                    }
                )
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break
        payload = {"query": query, "match_count": len(matches), "matches": matches}
        return ToolResult(content=json.dumps(payload, ensure_ascii=False, indent=2), metadata={"result": payload})
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            content=f"text_search failed: {exc}",
            is_error=True,
            metadata={"error_type": exc.__class__.__name__},
        )


def execute_json_get(input: Dict[str, Any], _ctx: ToolContext) -> ToolResult:
    try:
        path = str(input.get("path") or "").strip()
        if not path:
            raise ValueError("path is required")
        if "data" in input:
            data = input["data"]
        elif input.get("json_text"):
            data = json.loads(str(input["json_text"]))
        else:
            raise ValueError("json_text or data is required")
        value = _json_get(data, path)
        payload = {"path": path, "value": value}
        return ToolResult(content=json.dumps(payload, ensure_ascii=False, indent=2), metadata={"result": payload})
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            content=f"json_get failed: {exc}",
            is_error=True,
            metadata={"error_type": exc.__class__.__name__},
        )


def execute_table_summarize(input: Dict[str, Any], _ctx: ToolContext) -> ToolResult:
    try:
        rows = input.get("rows")
        if isinstance(rows, str):
            rows = json.loads(rows)
        if not isinstance(rows, list):
            raise ValueError("rows must be an array")
        fields = input.get("fields") or _infer_fields(rows)
        fields = [str(field) for field in fields]
        limit = max(1, min(int(input.get("limit") or 5), 20))
        numeric_stats = {}
        for field in fields:
            values = [row.get(field) for row in rows if isinstance(row, dict)]
            numbers = [float(value) for value in values if isinstance(value, (int, float))]
            if numbers:
                numeric_stats[field] = {
                    "min": min(numbers),
                    "max": max(numbers),
                    "avg": sum(numbers) / len(numbers),
                }
        payload = {
            "row_count": len(rows),
            "fields": fields,
            "numeric_stats": numeric_stats,
            "sample_rows": [
                {field: row.get(field) for field in fields}
                for row in rows[:limit]
                if isinstance(row, dict)
            ],
        }
        return ToolResult(content=json.dumps(payload, ensure_ascii=False, indent=2), metadata={"result": payload})
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            content=f"table_summarize failed: {exc}",
            is_error=True,
            metadata={"error_type": exc.__class__.__name__},
        )


def register_common_builtin_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        BuiltinTool(
            tool_name="calculate",
            tool_description="Safely evaluate a basic arithmetic expression and return the numeric result.",
            schema=CALCULATE_SCHEMA,
            handler=execute_calculate,
            tool_capabilities=ToolCapabilities(read=True),
            is_read_only=True,
        )
    )
    registry.register(
        BuiltinTool(
            tool_name="text_search",
            tool_description="Search provided in-memory documents and return matching snippets.",
            schema=TEXT_SEARCH_SCHEMA,
            handler=execute_text_search,
            tool_capabilities=ToolCapabilities(read=True),
            is_read_only=True,
        )
    )
    registry.register(
        BuiltinTool(
            tool_name="json_get",
            tool_description="Extract a value from a JSON document by a simple dot/bracket path.",
            schema=JSON_GET_SCHEMA,
            handler=execute_json_get,
            tool_capabilities=ToolCapabilities(read=True),
            is_read_only=True,
        )
    )
    registry.register(
        BuiltinTool(
            tool_name="table_summarize",
            tool_description="Summarize in-memory table rows with sample rows and numeric statistics.",
            schema=TABLE_SUMMARIZE_SCHEMA,
            handler=execute_table_summarize,
            tool_capabilities=ToolCapabilities(read=True),
            is_read_only=True,
        )
    )
    return registry


def build_common_builtin_tool_registry() -> ToolRegistry:
    return register_common_builtin_tools(ToolRegistry())


def _safe_eval_expression(expression: str) -> float:
    allowed_binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    allowed_unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("power exponent is too large")
            return float(allowed_binary[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return float(allowed_unary[type(node.op)](evaluate(node.operand)))
        raise ValueError(f"unsupported expression node: {node.__class__.__name__}")

    result = evaluate(ast.parse(expression, mode="eval"))
    if not math.isfinite(result):
        raise ValueError("result is not finite")
    return result


def _coerce_documents(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("documents must be an array")
    documents = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            documents.append(item)
        else:
            documents.append({"id": str(index), "text": str(item)})
    return documents


def _snippet(text: str, start: int, end: int, radius: int = 45) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].replace("\n", " ")


def _parse_json_path(path: str) -> List[Any]:
    path = path.strip()
    if path.startswith("$."):
        path = path[2:]
    elif path == "$":
        return []
    elif path.startswith("$"):
        path = path[1:]
    tokens: List[Any] = []
    for part in [part for part in path.split(".") if part]:
        while "[" in part:
            key, rest = part.split("[", 1)
            if key:
                tokens.append(key)
            index_text, part = rest.split("]", 1)
            tokens.append(int(index_text))
        if part:
            tokens.append(part)
    return tokens


def _json_get(data: Any, path: str) -> Any:
    current = data
    for token in _parse_json_path(path):
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise KeyError(f"array index not found: {token}")
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                raise KeyError(f"object key not found: {token}")
            current = current[token]
    return current


def _infer_fields(rows: List[Any]) -> List[str]:
    fields: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            if key not in fields:
                fields.append(str(key))
    return fields
