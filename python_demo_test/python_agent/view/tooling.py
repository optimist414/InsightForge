from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .prompt import commented_field, to_plain_dict


class ApprovalRequirement(str, Enum):
    """工具审批要求。当前 demo 不启用审批，只保留枚举供后续扩展。"""

    NEVER = "never"
    ON_REQUEST = "on_request"
    ALWAYS = "always"


class ToolErrorKind(str, Enum):
    """工具错误类型。"""

    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    EXECUTION_ERROR = "execution_error"
    APPROVAL_REQUIRED = "approval_required"
    MCP_ERROR = "mcp_error"
    SKILL_ERROR = "skill_error"


class ToolFailureCode(str, Enum):
    """面向 Agent 的稳定失败码，不暴露底层异常格式。"""

    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    NETWORK_UNAVAILABLE = "NETWORK_UNAVAILABLE"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    ACCESS_DENIED = "ACCESS_DENIED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    MCP_ERROR = "MCP_ERROR"
    SKILL_ERROR = "SKILL_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class ToolCapabilities:
    """工具能力边界，用于后续权限、风控和并行调度。"""

    read: bool = commented_field("是否具备读取本地或外部资源的能力。", False)
    write: bool = commented_field("是否具备写入文件、数据库或外部系统的能力。", False)
    network: bool = commented_field("是否需要网络访问。", False)
    shell: bool = commented_field("是否会触发 shell/命令执行。", False)
    requires_approval: bool = commented_field("是否默认需要用户审批。", False)


@dataclass
class ToolContext:
    """工具执行上下文。

    当前 demo 只需要默认值；后续可以由 arrange 层填入真实 session、
    turn、MCP pool、审批策略和事件发送器。
    """

    workspace_root: str = commented_field("当前工作区根目录。", "")
    session_id: str = commented_field("当前 agent 会话 ID。", "")
    turn_id: str = commented_field("当前用户轮次 ID。", "")
    tx_event: Any = commented_field("工具执行事件发送器或回调；当前默认为 None。", None)
    mcp_pool: Any = commented_field("MCP 连接池或客户端集合；当前默认为 None。", None)
    approval_policy: ApprovalRequirement = commented_field("当前执行审批策略。", ApprovalRequirement.NEVER)


@dataclass
class ToolResult:
    """工具统一返回结果。"""

    content: str = commented_field("工具返回给 agent/模型看的文本内容。", "")
    is_error: bool = commented_field("本次工具调用是否失败。", False)
    metadata: Optional[Dict[str, Any]] = commented_field("结构化元数据，供 loop 记录或调试使用。", None)
    failure: Optional["ToolFailureDTO"] = commented_field(
        "失败时的统一失败 DTO；成功时为 None。",
        None,
    )


@dataclass
class ToolFailureDTO:
    """工具失败的统一返回 DTO。

    所有工具失败都经由该对象返回给 Engine 和模型。`message` 是可展示的
    简短原因，底层异常栈不会进入模型上下文；模型应优先依据 `error_code`、
    `retryable` 和 `suggested_action` 决定重试、换参或切换数据源。
    """

    success: bool = commented_field("固定为 false，便于模型和调用方快速判断。", False)
    tool_name: str = commented_field("发生失败的工具名称。", "")
    error_code: ToolFailureCode = commented_field("稳定的机器可读失败码。", ToolFailureCode.INTERNAL_ERROR)
    message: str = commented_field("可展示的简短失败原因。", "")
    retryable: bool = commented_field("当前输入不变时是否值得稍后重试。", False)
    failed_stage: str = commented_field("失败阶段，例如 validation、execution、dependency。", "execution")
    suggested_action: str = commented_field("给 Agent 的下一步建议，例如换数据源或修正参数。", "")
    retry_after_ms: Optional[int] = commented_field("建议等待后重试的毫秒数；未知时为 None。", None)
    cause_type: Optional[str] = commented_field("底层异常类名，仅用于日志和调试。", None)

    def to_payload(self) -> Dict[str, Any]:
        """生成发送给模型的固定 JSON 结构。"""

        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "failure": {
                "error_code": self.error_code.value,
                "message": self.message,
                "retryable": self.retryable,
                "failed_stage": self.failed_stage,
                "suggested_action": self.suggested_action,
                "retry_after_ms": self.retry_after_ms,
            },
        }

    def to_result(self) -> ToolResult:
        """适配现有 ToolResult 外壳，保持 Chat Completions tool 消息兼容。"""

        payload = self.to_payload()
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            is_error=True,
            metadata={"result": payload, "failure": payload["failure"], "cause_type": self.cause_type},
            failure=self,
        )

    @classmethod
    def from_exception(
        cls,
        tool_name: str,
        exc: Exception,
        *,
        failed_stage: str = "execution",
    ) -> "ToolFailureDTO":
        """将任意旧工具异常收敛成稳定错误码。"""

        message = str(exc).strip() or exc.__class__.__name__
        lower_message = message.lower()
        error_code = ToolFailureCode.INTERNAL_ERROR
        retryable = False
        suggested_action = "检查工具输入和运行日志后再继续。"

        if isinstance(exc, (ValueError, TypeError, KeyError, json.JSONDecodeError)):
            error_code = ToolFailureCode.INVALID_INPUT
            failed_stage = "validation"
            suggested_action = "检查并修正工具参数后重试。"
        elif isinstance(exc, TimeoutError) or "timeout" in lower_message or "timed out" in lower_message:
            error_code = ToolFailureCode.TIMEOUT
            retryable = True
            suggested_action = "稍后重试，或缩小查询范围、降低返回量。"
        elif isinstance(exc, PermissionError) or "access denied" in lower_message or "forbidden" in lower_message:
            error_code = ToolFailureCode.ACCESS_DENIED
            suggested_action = "检查访问权限、账号状态或授权配置。"
        elif "rate limit" in lower_message or "too many requests" in lower_message or "http 429" in lower_message:
            error_code = ToolFailureCode.RATE_LIMITED
            retryable = True
            suggested_action = "等待限流窗口结束后重试，避免并发调用。"
        elif (
            isinstance(exc, (ConnectionError, OSError))
            or "connection refused" in lower_message
            or "network" in lower_message
        ):
            error_code = ToolFailureCode.NETWORK_UNAVAILABLE
            retryable = True
            failed_stage = "dependency"
            suggested_action = "检查网络、代理和外部服务可用性，或切换数据源。"
        elif "not found" in lower_message:
            error_code = ToolFailureCode.TOOL_NOT_FOUND
            suggested_action = "确认工具名称或选择可用的替代工具。"
        elif (
            "edge cdp" in lower_message
            or "microsoft edge was not found" in lower_message
            or "browser websocket is not connected" in lower_message
        ):
            error_code = ToolFailureCode.DEPENDENCY_UNAVAILABLE
            retryable = True
            failed_stage = "dependency"
            suggested_action = "检查浏览器适配器；可改用 API、联网搜索或其他数据源。"
        elif "no video cards" in lower_message or "no data" in lower_message:
            error_code = ToolFailureCode.DATA_UNAVAILABLE
            retryable = True
            suggested_action = "换用其他公开数据源，或稍后重试该页面。"
        elif "extract" in lower_message:
            error_code = ToolFailureCode.EXTRACTION_FAILED
            suggested_action = "换一个链接、降低正文要求，或使用浏览器类工具。"

        return cls(
            tool_name=tool_name,
            error_code=error_code,
            message=message,
            retryable=retryable,
            failed_stage=failed_stage,
            suggested_action=suggested_action,
            cause_type=exc.__class__.__name__,
        )

    @classmethod
    def from_legacy_result(cls, tool_name: str, result: ToolResult) -> "ToolFailureDTO":
        """兼容尚未迁移的旧 handler 返回的 is_error=True 结果。"""

        metadata = result.metadata or {}
        existing = metadata.get("failure")
        if isinstance(existing, dict):
            code_text = str(existing.get("error_code") or ToolFailureCode.INTERNAL_ERROR.value)
            try:
                code = ToolFailureCode(code_text)
            except ValueError:
                code = ToolFailureCode.INTERNAL_ERROR
            return cls(
                tool_name=str(metadata.get("tool_name") or tool_name),
                error_code=code,
                message=str(existing.get("message") or result.content or "Tool execution failed"),
                retryable=bool(existing.get("retryable", False)),
                failed_stage=str(existing.get("failed_stage") or "execution"),
                suggested_action=str(existing.get("suggested_action") or "检查工具日志后继续。"),
                retry_after_ms=existing.get("retry_after_ms"),
                cause_type=metadata.get("cause_type") or metadata.get("error_type"),
            )
        legacy_error = RuntimeError(str(result.content or "Tool execution failed"))
        return cls.from_exception(tool_name, legacy_error)


@dataclass
class ToolError:
    """工具错误对象，可转换为 ToolResult。"""

    message: str = commented_field("错误信息。", "")
    kind: ToolErrorKind = commented_field("错误类型。", ToolErrorKind.EXECUTION_ERROR)

    def to_result(self, tool_name: str = "") -> ToolResult:
        """兼容旧接口，并转换为统一失败 DTO。"""

        code_by_kind = {
            ToolErrorKind.NOT_FOUND: ToolFailureCode.TOOL_NOT_FOUND,
            ToolErrorKind.INVALID_INPUT: ToolFailureCode.INVALID_INPUT,
            ToolErrorKind.MCP_ERROR: ToolFailureCode.MCP_ERROR,
            ToolErrorKind.SKILL_ERROR: ToolFailureCode.SKILL_ERROR,
        }
        return ToolFailureDTO(
            tool_name=tool_name,
            error_code=code_by_kind.get(self.kind, ToolFailureCode.INTERNAL_ERROR),
            message=self.message,
            retryable=self.kind in {ToolErrorKind.MCP_ERROR},
            failed_stage="validation" if self.kind == ToolErrorKind.INVALID_INPUT else "execution",
            suggested_action="检查工具名称和参数后重试。"
            if self.kind in {ToolErrorKind.NOT_FOUND, ToolErrorKind.INVALID_INPUT}
            else "检查外部依赖或选择替代工具。",
            cause_type=self.kind.value,
        ).to_result()


@dataclass
class ApiToolSchema:
    """Provider function-calling 风格工具 schema。"""

    type: str = commented_field("API 工具类型；当前固定为 function。", "function")
    function_name: str = commented_field("function.name，对应内部工具名。", "")
    function_description: str = commented_field("function.description，给模型看的工具说明。", "")
    function_parameters: Dict[str, Any] = commented_field("function.parameters，JSON Schema 参数定义。", default_factory=dict)
    function_strict: bool = commented_field("function.strict，是否要求严格遵循参数 schema。", False)

    def to_provider_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "function": {
                "name": self.function_name,
                "description": self.function_description,
                "parameters": self.function_parameters or {"type": "object", "properties": {}},
                "strict": self.function_strict,
            },
        }


class ToolSpec:
    """工具统一接口。

    Python 里不强制用 interface 关键字，这里用基类表达图里的接口。
    子类需要实现 `execute`，其他字段通过方法暴露给 registry 和 prompt。
    """

    def name(self) -> str:
        raise NotImplementedError

    def description(self) -> str:
        raise NotImplementedError

    def input_schema(self) -> Dict[str, Any]:
        raise NotImplementedError

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities()

    def approval_requirement(self) -> ApprovalRequirement:
        return ApprovalRequirement.NEVER

    def read_only(self) -> bool:
        caps = self.capabilities()
        return caps.read and not caps.write and not caps.shell

    def supports_parallel(self) -> bool:
        return True

    def model_visible(self) -> bool:
        return True

    def defer_loading(self) -> bool:
        return False

    def api_tool_schema(self) -> ApiToolSchema:
        return ApiToolSchema(
            function_name=self.name(),
            function_description=self.description(),
            function_parameters=self.input_schema(),
            function_strict=False,
        )

    def execute(self, input: Dict[str, Any], ctx: Optional[ToolContext] = None) -> ToolResult:
        raise NotImplementedError


ToolHandler = Callable[[Dict[str, Any], ToolContext], ToolResult]


@dataclass
class BuiltinTool(ToolSpec):
    """内置工具实现。"""

    tool_name: str = commented_field("工具名称。", "")
    tool_description: str = commented_field("工具说明。", "")
    schema: Dict[str, Any] = commented_field("工具输入 JSON Schema。", default_factory=dict)
    handler: Optional[ToolHandler] = commented_field("工具执行函数。", None)
    tool_capabilities: ToolCapabilities = commented_field("工具能力边界。", default_factory=ToolCapabilities)
    approval: ApprovalRequirement = commented_field("工具审批要求。", ApprovalRequirement.NEVER)
    is_read_only: bool = commented_field("是否只读。", True)
    parallel: bool = commented_field("是否支持并行执行。", True)
    visible_to_model: bool = commented_field("是否暴露给模型。", True)
    should_defer_loading: bool = commented_field("是否延迟加载 schema 或说明。", False)

    def name(self) -> str:
        return self.tool_name

    def description(self) -> str:
        return self.tool_description

    def input_schema(self) -> Dict[str, Any]:
        return self.schema

    def capabilities(self) -> ToolCapabilities:
        return self.tool_capabilities

    def approval_requirement(self) -> ApprovalRequirement:
        return self.approval

    def read_only(self) -> bool:
        return self.is_read_only

    def supports_parallel(self) -> bool:
        return self.parallel

    def model_visible(self) -> bool:
        return self.visible_to_model

    def defer_loading(self) -> bool:
        return self.should_defer_loading

    def execute(self, input: Dict[str, Any], ctx: Optional[ToolContext] = None) -> ToolResult:
        if self.handler is None:
            return ToolError(
                message=f"Builtin tool has no handler: {self.tool_name}",
                kind=ToolErrorKind.EXECUTION_ERROR,
            ).to_result(self.tool_name)
        return self.handler(input, ctx or ToolContext())


@dataclass
class McpToolAdapter(ToolSpec):
    """MCP 工具适配器。当前仅保留字段和默认执行占位。"""

    server_name: str = commented_field("MCP server 名称。", "")
    tool_name: str = commented_field("MCP tool 名称。", "")
    schema: Dict[str, Any] = commented_field("MCP tool 输入 schema。", default_factory=dict)
    tool_description: str = commented_field("MCP tool 描述。", "")
    visible_to_model: bool = commented_field("是否暴露给模型。", True)

    def name(self) -> str:
        return f"{self.server_name}.{self.tool_name}" if self.server_name else self.tool_name

    def description(self) -> str:
        return self.tool_description or f"MCP tool {self.tool_name} from {self.server_name}."

    def input_schema(self) -> Dict[str, Any]:
        return self.schema

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(network=True)

    def model_visible(self) -> bool:
        return self.visible_to_model

    def execute(self, input: Dict[str, Any], ctx: Optional[ToolContext] = None) -> ToolResult:
        return ToolError(
            message="MCP execution is not wired in this view-layer demo.",
            kind=ToolErrorKind.MCP_ERROR,
        ).to_result(self.name())


@dataclass
class SkillTool(ToolSpec):
    """Skill 工具适配器。当前保留 skill 文件读取入口。"""

    skill_name: str = commented_field("skill 名称。", "")
    skill_path: str = commented_field("skill 文件或目录路径。", "")
    schema: Dict[str, Any] = commented_field("skill 工具输入 schema。", default_factory=dict)
    tool_description: str = commented_field("skill 能力描述。", "")
    visible_to_model: bool = commented_field("是否暴露给模型。", True)

    def name(self) -> str:
        return self.skill_name

    def description(self) -> str:
        return self.tool_description or f"Skill tool {self.skill_name}."

    def input_schema(self) -> Dict[str, Any]:
        return self.schema

    def load_skill(self) -> ToolResult:
        return ToolResult(
            content=f"Skill loading placeholder: {self.skill_path}",
            is_error=False,
            metadata={"skill_name": self.skill_name, "skill_path": self.skill_path},
        )

    def read_skill_file(self) -> ToolResult:
        return self.load_skill()

    def execute(self, input: Dict[str, Any], ctx: Optional[ToolContext] = None) -> ToolResult:
        return ToolError(
            message="Skill execution is not wired in this view-layer demo.",
            kind=ToolErrorKind.SKILL_ERROR,
        ).to_result(self.name())


@dataclass
class ToolRegistry:
    """工具注册表。"""

    tools: Dict[str, ToolSpec] = commented_field("工具名到 ToolSpec 的映射。", default_factory=dict)

    def register(self, tool: ToolSpec) -> None:
        self.tools[tool.name()] = tool

    def get(self, name: str) -> Optional[ToolSpec]:
        return self.tools.get(name)

    def list(self) -> List[ToolSpec]:
        return list(self.tools.values())

    def list_api_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            tool.api_tool_schema().to_provider_dict()
            for tool in self.list()
            if tool.model_visible() and not tool.defer_loading()
        ]

    def execute_full_with_context(
        self,
        name: str,
        input: Dict[str, Any],
        ctx: Optional[ToolContext] = None,
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolError(
                message=f"Tool not found: {name}",
                kind=ToolErrorKind.NOT_FOUND,
            ).to_result(name)
        try:
            result = tool.execute(input, ctx or ToolContext())
            if not isinstance(result, ToolResult):
                raise TypeError(f"Tool {name} returned {type(result).__name__}, expected ToolResult")
        except Exception as exc:  # noqa: BLE001 - registry is the final tool execution boundary.
            return ToolFailureDTO.from_exception(name, exc).to_result()
        if not result.is_error:
            return result
        if result.failure is not None:
            return result
        return ToolFailureDTO.from_legacy_result(name, result).to_result()


def tool_field_comments() -> Dict[str, Dict[str, str]]:
    """导出工具相关 dataclass 字段中文注释。"""

    classes = [
        ToolCapabilities,
        ToolContext,
        ToolResult,
        ToolFailureDTO,
        ToolError,
        ApiToolSchema,
        BuiltinTool,
        McpToolAdapter,
        SkillTool,
        ToolRegistry,
    ]
    return {
        cls.__name__: {
            name: field_info.metadata.get("comment", "")
            for name, field_info in cls.__dataclass_fields__.items()
        }
        for cls in classes
    }


def build_demo_tool_registry() -> ToolRegistry:
    """Build a tiny registry used by the schema-to-call demo."""

    registry = ToolRegistry()
    registry.register(
        BuiltinTool(
            tool_name="echo",
            tool_description="返回输入文本，用于验证工具 schema 获取和调用链路。",
            schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要原样返回的文本。",
                    }
                },
            },
            handler=lambda input, ctx: ToolResult(
                content=str(input.get("text", "")),
                metadata={"workspace_root": ctx.workspace_root},
            ),
        )
    )
    return registry


def run_tool_schema_call_demo() -> Dict[str, Any]:
    """Simulate: registry -> schema visible to agent -> tool call -> result."""

    registry = build_demo_tool_registry()
    schemas = registry.list_api_tool_schemas()
    selected_schema = schemas[0]
    tool_name = selected_schema["function"]["name"]
    tool_input = {"text": "hello tool schema"}
    result = registry.execute_full_with_context(
        name=tool_name,
        input=tool_input,
        ctx=ToolContext(workspace_root="demo-workspace", session_id="session_demo", turn_id="turn_demo"),
    )
    return {
        "agent_visible_schemas": schemas,
        "agent_selected_tool": {
            "name": tool_name,
            "input": tool_input,
        },
        "tool_result": to_plain_dict(result),
        "field_comments": tool_field_comments(),
    }


def main() -> None:
    print(json.dumps(run_tool_schema_call_demo(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
