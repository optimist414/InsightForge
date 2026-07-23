# Agent 输入管理框架 MVP 技术文档

## 1. 总体数据流

```text
UserInput
-> Goal
-> MemoryStore.select_for_agent_input()
-> AgentInput with empty LoopContext
-> ReAct Loop reads prior LoopContextSummary at each loop start
-> Agent emits Thought+Action protocol JSON
-> AgentStepOutputParser parses Action
-> ToolExecutionLayer executes local/MCP/skill tools
-> LoopViewUpdater updates loop view
-> LoopContext events / observations / errors / state / loop memories
-> AgentOutput
-> FinalResponse
```

实现入口：

- `ReactAgentMvp.build_input()`
- `ReactAgentMvp.run()`
- `DeepSeekReactPlanner.generate_step_output()`

核心数据类位于 `view/models.py`。旧的顶层实现文件已删除，新实现统一使用分层包导入。

提示词请求结构位于 `view/prompt.py`。它把一次 LLM 调用从“拼接字符串”改成内部标准对象：

```text
PromptRequest
-> SystemPrompt
-> MessageHistory
-> ToolCatalog
-> ProviderChatRequest
```

`DeepSeekReactPlanner` 不再直接手写 system/user prompt，而是调用 `ReactPromptViewBuilder` 构造 `PromptRequest`，再转换为 provider chat messages。

## 1.1 代码分层

| 层 | 路径 | 职责 | 典型对象 |
| --- | --- | --- | --- |
| view | `view/` | 定义 Agent 输入、loop 视图、输出视图，以及更新 loop 视图的写入器 | `AgentInput`, `LoopContext`, `AgentOutput`, `LoopViewUpdater` |
| arrange | `arrange/` | 编排单任务 ReAct loop，把 view、planner、tools、external_info、memory 串起来 | `ReactAgentMvp` |
| external_info | `external_info/` | 外部工具和外部知识入口，内部可继续细分 MCP、skill 等 | `ToolRegistry`, `ToolExecutionLayer`, `demo_mcp_reference` |
| memory | `memory/` | 记忆读取和写入入口；当前阶段长短期记忆置空，只保留接口 | `InMemoryMemoryStore` |
| planner | `planner/` | 生成 Thought+Action 协议，可替换成规则、DeepSeek 或其他模型 | `DeepSeekReactPlanner`, `DeepSeekChatClient` |
| tools | `tools/` | 辅助编排工具类，不直接代表外部世界；当前包括 agent 输出协议解析 | `AgentStepOutputParser` |

分层依赖方向：

```text
arrange
-> view
-> memory
-> planner
-> tools
-> external_info

planner -> view
tools -> view
external_info -> view
memory -> view
```

其中 `external_info` 表示“外部可调用信息源”，例如 MCP、skill、HTTP 工具、数据库工具；`tools` 表示“编排内部辅助工具”，例如协议解析器、格式修复器、schema 校验器。

当前复杂 MCP 工具：

| 工具 | 输入 | 输出摘要 |
| --- | --- | --- |
| `mcp_requirement_analyzer` | `{"task": "用户原始任务"}` | 子任务、推荐后续工具、完成标准 |
| `mcp_hot_news_schema` | `{"focus": "需要的库表信息"}` | `hot_news`、`news_tag`、`hot_news_tag` 字段与 join 规则 |
| `mcp_platform_policy` | `{"platforms": ["知乎", "微博"]}` | 平台代码、URL 规则、标签关键词 |
| `mcp_query_plan_quality_check` | `{"plan": "查询方案草案"}` | 时间、平台、标签、字段、排序等质检清单 |

## 1.2 提示词请求架构

提示词模块对应 `view/prompt.py`，职责是把面向 agent 的 view 转成一次 LLM 请求。核心类如下：

| 类 | 作用 |
| --- | --- |
| `PromptRequest` | 内部统一请求对象，聚合模型、system、messages、tools、tool_choice、stream、thinking。 |
| `SystemPrompt` | 将 system prompt 拆成 `stable_prefix`、`dynamic_context`、`runtime_instructions`。 |
| `MessageHistory` | 管理 user/assistant/tool 消息追加，并保留 turn metadata。 |
| `Message` | 内部消息基类，消息内容由多个 `ContentBlock` 组成。 |
| `UserMessage` | 用户消息，保留 `raw_user_text` 和 `TurnMetadata`。 |
| `AssistantMessage` | assistant 消息，保留可展示文本、thinking 和 tool uses。 |
| `ToolResultMessage` | 工具结果消息，使用 `tool_use_id` 和 assistant 工具调用对齐。 |
| `ContentBlock` | 文本、thinking、tool use、tool result、image url、server tool use 的统一内容块。 |
| `TurnMetadata` | 当前日期、工作区、模型、模式、权限、输入来源、资源元数据、工作集摘要。 |
| `ToolCatalog` | 本轮 prompt 可见工具列表和 tool choice。 |
| `Tool` | 工具名、描述、输入 schema、strict、defer_loading。 |
| `ProviderChatRequest` | provider 适配层，包含 model/messages/tools/tool_choice/stream。 |
| `ReactPromptViewBuilder` | 把 `AgentInput`、`LoopContextSummary`、`LoopOutput`、工具目录组装成 `PromptRequest`。 |

字段注释：

- 每个 dataclass 字段使用 `metadata={"comment": "..."}` 保存中文注释。
- `prompt_field_comments()` 可以导出字段注释，后续可用于 schema、调试页面或文档生成。

提示词拆分规则：

| SystemPrompt 字段 | 内容 |
| --- | --- |
| `stable_prefix` | 稳定协议，例如 ReAct planner 身份、Thought+Action JSON 输出格式。 |
| `dynamic_context` | 动态输入，例如目标、记忆、loop 摘要、可用工具目录。 |
| `runtime_instructions` | 运行时决策规则，例如何时调用 calculator/MCP，何时 final_response。 |

当前 DeepSeek 路径：

```text
DeepSeekReactPlanner.generate_step_output()
-> ReactPromptViewBuilder.build()
-> PromptRequest.to_provider_chat_request()
-> DeepSeekChatClient.chat(provider_request.messages)
```

当前仍然让 DeepSeek 输出 JSON 协议文本，不直接使用 provider tool-call 机制。`ProviderChatRequest.tools` 已保留为后续接入真正 function calling 的适配点。

### 1.2.1 Chat Completions Adapter

实际输入给 LLM 的不是工程对象，而是 Chat Completions 风格 JSON。适配器位于：

- `view/chat_completions_adapter.py`

职责：

```text
PromptRequest / MessageHistory / ToolCatalog
-> ChatCompletionsAdapter
-> {
     "model": "...",
     "messages": [...],
     "tools": [...],
     "tool_choice": "auto",
     "stream": true
   }
```

转换约束：

- `SystemPrompt` 三段内容只拼成 `messages[0] = {"role": "system", "content": "..."}`。
- 当前 `UserMessage` 会把用户输入和非空 `TurnMetadata` 拼成一条 `role=user` 消息。
- 历史 `MessageHistory` 按顺序转成 `user / assistant / tool` 消息。
- 工具注册表或工具目录只转成顶层 `tools` 字段，不塞进 system/user content 主路径。
- `tool_choice` 只在存在工具时写入请求。
- `stream` 直接来自 `PromptRequest.stream`。

消息转换：

| 内部对象 | Chat Completions 格式 |
| --- | --- |
| `SystemPrompt` | `{"role": "system", "content": "..."}` |
| `UserMessage` | `{"role": "user", "content": "..."}` |
| `AssistantMessage(text)` | `{"role": "assistant", "content": "..."}` |
| `AssistantMessage(tool_uses)` | `{"role": "assistant", "content": null, "tool_calls": [...]}` |
| `ToolResultMessage` / `ToolResult` | `{"role": "tool", "tool_call_id": "...", "content": "..."}` |
| `ToolCatalog` / `ToolRegistry` | `tools: [{"type": "function", "function": {...}}]` |

Demo：

```bash
python -m python_demo_test.python_agent.testpy.run_chat_completions_adapter_demo \
  --output python_demo_test/agent_input_framework_mvp/output/chat_completions_adapter_demo.json
```

该 demo 会展示：

1. 工程 `PromptRequest` 转成 Chat Completions request。
2. 模拟 assistant 返回 `tool_calls`。
3. adapter 将 `tool_calls` 解析成内部 `ToolUse`。
4. `ToolRegistry` 执行工具。
5. 工具结果转成 `role=tool` 消息，供下一轮 LLM 输入。

## 1.3 工具视图架构

工具视图位于 `view/tooling.py`，先按完整工具程序图保留字段和扩展点，但当前 demo 只使用最小链路：

```text
ToolRegistry
-> ToolSpec.name / description / input_schema
-> ApiToolSchema
-> agent 选择工具名和参数
-> ToolRegistry.execute_full_with_context()
-> ToolResult
```

核心对象：

| 类 | 作用 |
| --- | --- |
| `ToolSpec` | 工具统一接口，定义名称、描述、输入 schema、能力、审批、可见性、执行等方法。 |
| `ToolRegistry` | 工具注册表，支持 register/get/list/execute_full_with_context。 |
| `ToolContext` | 工具执行上下文，预留 workspace、session、turn、MCP pool、审批策略等字段。 |
| `ToolResult` | 工具统一返回，包含 content、is_error、metadata。 |
| `ToolError` | 工具错误对象，可转换为 `ToolResult`。 |
| `ToolCapabilities` | 工具能力边界，预留 read/write/network/shell/requires_approval。 |
| `ApprovalRequirement` | 审批策略枚举：Never/OnRequest/Always。 |
| `BuiltinTool` | 内置工具实现。 |
| `McpToolAdapter` | MCP 工具适配器，占位保留 server/tool/schema。 |
| `SkillTool` | Skill 工具适配器，占位保留 skill 文件加载入口。 |
| `ApiToolSchema` | provider function-calling 风格工具 schema。 |

字段注释：

- 工具相关 dataclass 字段同样使用 `metadata={"comment": "..."}` 保存中文注释。
- `tool_field_comments()` 可导出字段注释。

Demo：

```bash
python -m python_demo_test.python_agent.testpy.run_tool_schema_demo \
  --output python_demo_test/agent_input_framework_mvp/output/tool_schema_demo.json
```

当前 demo 注册一个 `echo` 内置工具，输出：

- `agent_visible_schemas`: agent 可见的 function schema。
- `agent_selected_tool`: 模拟 agent 根据 schema 选择的工具和参数。
- `tool_result`: registry 带 `ToolContext` 执行后的结果。
- `field_comments`: 工具字段中文注释。

## 1.4 Engine Loop 架构

Engine loop 位于 `engine/core.py`，对应新的运行流程图。它不再依赖旧 `view/models.py` 和 `loop_view.py`，而是直接使用：

- `view.prompt`: `MessageHistory`, `AssistantMessage`, `ToolUse`, `ToolResultMessage`
- `view.tooling`: `ToolRegistry`, `ToolContext`, `ToolResult`
- `view.chat_completions_adapter`: Chat Completions 请求和消息转换

核心对象：

| 类 | 作用 |
| --- | --- |
| `Engine` | turn 总调度器，处理用户消息、刷新 system prompt、运行 loop、执行工具、写入 session。 |
| `Turn` | 单轮执行状态，记录 step、usage、tool calls。 |
| `Session` | 会话状态，保存 messages、system_prompt、working_set、settings。 |
| `LlmClient` | LLM 流式接口抽象。 |
| `ScriptedLlmClient` | demo LLM，第一步返回 tool call，第二步返回最终文本。 |
| `ChatCompletionsSseClient` | OpenAI-compatible Chat Completions SSE client，消费真实 `data:` 流并转换为 `StreamEvent`。 |
| `MessageRequestBuilder` | 从 session 和 tool registry 构造 Chat Completions 请求。 |
| `StreamEvent` | 流式事件对象，覆盖 MessageStart、ContentBlockDelta、MessageStop 等。 |
| `ToolUseState` | 收集流式 tool call 的 name、id 和 input buffer，并解析 JSON input。 |
| `ToolExecutionPlan` | 工具执行计划，预留 approval、read_only、parallel_group。 |
| `ToolExecutionOutcome` | 工具执行结果，包含 tool_use_id、ToolResult、耗时和错误状态。 |
| `CoreEvent` | Engine 对外事件，包含 TurnStarted、ToolCallComplete、TurnComplete 等。 |
| `McpPool` | MCP 调用占位。 |
| `StopController` | 停止条件抽象。 |
| `MaxStepsStopController` | 当前实现：按最大 step 数停止。 |

当前 loop 流程：

```text
Engine.handle_send_message()
-> Session.messages += UserMessage
-> refresh_system_prompt()
-> run_turn_loop()
   -> MessageRequestBuilder.build()
   -> LlmClient.create_message_stream()
   -> StreamEvent tool_use delta -> ToolUseState
   -> ToolExecutionPlan
   -> execute_tool_with_lock()
   -> ToolRegistry.execute_full_with_context()
   -> Session.messages += AssistantMessage(tool_calls)
   -> Session.messages += ToolResultMessage
   -> 下一 step
   -> LlmClient 返回 final text
   -> Session.messages += AssistantMessage(text)
-> CoreEvent.TurnComplete
```

停止条件：

```text
StopController.should_stop(turn, session, events)
```

当前只实现 `MaxStepsStopController(max_steps=n)`，后续可扩展为：

- 无 pending tool call 后停止。
- 达到 token 预算后停止。
- 工具连续失败后停止。
- 用户中断后停止。

Demo：

```bash
python -m python_demo_test.python_agent.testpy.run_engine_loop_demo \
  --output python_demo_test/agent_input_framework_mvp/output/engine_loop_demo.json \
  --max-steps 4
```

验证结果：

```text
session messages: user -> assistant(tool_calls) -> tool -> assistant(text)
core events: TurnStarted -> ToolCallStarted -> ToolCallComplete -> MessageComplete -> TurnComplete
```

## 1.5 Arrange 多 Turn 编排

`arrange/multi_turn_agent.py` 负责真正的多 turn 外层编排。这里的多 turn 指同一个 `Session` 生命周期里，可以连续接收多次用户对话；每次用户输入都会追加到同一个会话历史中，再启动一个新的单 turn engine loop。

它不再直接解析旧 ReAct 文本协议，也不再依赖已删除的 `view/models.py`、`view/loop_view.py`，而是把每个用户 turn 委托给 `engine/core.py` 中的 `Engine`。

边界划分：

| 层 | 职责 |
| --- | --- |
| `MultiTurnAgent` | 外层会话编排，持有同一个 `Session`，通过 `send_message()` 连续接收用户消息。 |
| `Engine` | 单个用户 turn 的内部 loop，负责构造 Chat Completions 请求、消费 SSE/流事件、执行工具、写回消息。 |
| `StopController` | 每个 turn 的停止条件。`MultiTurnAgent` 每一轮都会通过 factory 创建新的 controller。 |

当前实现：

```text
MultiTurnAgent.send_message(user_text)
-> turn_index += 1
-> 复用 shared_session
-> turn_stop_controller_factory(turn_index, user_text)
-> Engine(shared_session, llm_client, tool_registry, stop_controller)
-> Engine.handle_send_message(user_text)
-> 返回 AgentTurnResult
```

关键点：

- 多轮共享同一个 `Session`，因此历史 user / assistant / tool 消息会进入下一轮请求。
- 每个 turn 独立创建 `StopController`，当前默认是 `MaxStepsStopController(max_steps=12)`。
- 外层会话最多接收多少次用户输入由 `max_turns` 控制；内层单 turn step 停止由 `StopController` 控制。
- `run_turns()` 只是 demo/批量测试辅助，本质上是循环调用 `send_message()`。
- demo LLM 已按“当前 user message 之后是否已有 tool result”判断当前 turn 状态，避免第二轮误读第一轮工具结果。

Demo：

```bash
python -m python_demo_test.python_agent.testpy.run_arrange_n_turn_demo \
  --turns 3 \
  --max-steps-per-turn 4 \
  --output python_demo_test/agent_input_framework_mvp/output/arrange_n_turn_demo.json
```

验证结果：

```text
executed_turn_count: 3
session messages: 12
每个 turn: user -> assistant(tool_calls) -> tool -> assistant(text)
每个 turn stop_controller: MaxStepsStopController
```

真实 SSE Demo：

```bash
python -m python_demo_test.python_agent.testpy.run_sse_live_demo \
  --output python_demo_test/agent_input_framework_mvp/output/sse_live_demo.json \
  --max-steps 2
```

工具调用 SSE Demo：

```bash
python -m python_demo_test.python_agent.testpy.run_sse_live_demo \
  --with-tools \
  --user-input "请调用 echo 工具，参数 text 为 live tool test。" \
  --output python_demo_test/agent_input_framework_mvp/output/sse_live_tool_demo.json \
  --max-steps 3
```

当前已验证：

- 普通文本 SSE：`delta.content` 被转换成 `CoreEvent.MessageDelta`。
- 工具调用 SSE：`delta.tool_calls[].function.arguments` 分片被 `ToolUseState.input_buffer` 接收并解析。
- 工具结果写回：`assistant(tool_calls) -> tool -> assistant(text)`。
- `SSLKEYLOGFILE` 不可写时，SSE client 会临时移除该环境变量，避免 SSL 握手前失败。

调整后的关键点：

```text
接收用户输入
-> 生成 Goal
-> 读取短期/长期记忆
-> 组装 AgentInput，此时 LoopContext 为空
-> 进入 ReAct loop
   -> 每轮开始读取历史 LoopContextSummary
   -> 生成 Thought+Action 协议文本
   -> 解析协议文本为 Action
   -> 通过工具执行层调用本地工具、MCP 或 skill
   -> 写入本轮 observation/error/external reference/state/loop memory
   -> 下一轮再读取上一轮沉淀出的过程摘要
```

## 2. 流程图节点输入输出

### 2.1 接收用户输入

输入：

```python
user_input: str
```

输出：

```python
Goal(
    user_input=user_input,
    task_description="处理用户输入并通过 ReAct loop 生成可追踪输出：...",
    constraints=[...]
)
```

实现：

- `ReactAgentMvp._describe_task()`
- `ReactAgentMvp.build_input()`

### 2.2 生成目标 / 任务

输入：

- `user_input`

输出：

- `Goal`

字段：

| 字段 | 含义 |
| --- | --- |
| `goal_id` | 目标 ID |
| `user_input` | 用户原始输入 |
| `task_description` | 面向 Agent 的任务描述 |
| `constraints` | 本轮任务约束 |
| `metadata` | 扩展字段 |

### 2.3 读取短期记忆

输入：

- `MemoryStore`
- `MemoryType.SHORT_TERM`

输出：

```python
List[ShortTermMemory]
```

实现：

- `InMemoryMemoryStore.list_by_type()`
- `InMemoryMemoryStore.select_for_agent_input()`

短期记忆用于最近 n 轮摘要，MVP 用 `priority` 和 `created_at` 排序取前几条。

### 2.4 读取长期记忆

输入：

- `MemoryStore`
- `MemoryType.LONG_TERM`

输出：

```python
List[LongTermMemory]
```

长期记忆用于稳定偏好、固定约束、长期复用配置。MVP 用 `stable_key` 预留未来去重和更新入口。

### 2.5 初始化 LoopContext

在每次接收新的用户输入并组装 `AgentInput` 时，`loop_context` 必须是空的。

输入：

- 新建的 `AgentInput`

输出：

```python
LoopContext(
    tool_observations=[],
    errors=[],
    external_refs=[],
    loop_memories=[],
    events=[]
)
```

实现：

- `AgentInput.loop_context` 默认由 `LoopContext()` 创建。
- `ReactAgentMvp.build_input()` 只组装目标、短期/长期记忆、外部参考，不读取历史 loop 过程信息。

这样可以保证第一次 loop 不会误读尚未发生的过程信息。

### 2.6 在 Loop 中读取历史 Loop 过程信息

输入：

- `LoopContext`
- `current_loop_index`

输出：

```python
LoopContextSummary(
    current_loop_index=2,
    prior_loop_count=1,
    summary_text="prior_loops=1; observations=1; errors=0; loop_memories=1; ...",
    is_empty=False,
    tool_observation_count=1,
    error_count=0,
    external_reference_count=1,
    loop_memory_count=1,
    latest_state="Calling tool: calculator.",
    loop_memory_summaries=[...],
    external_reference_summaries=[...],
    event_summaries=[...]
)
```

实现：

- `ReactAgentMvp._read_loop_context_summary()`
- `LoopOutput.context_summary`
- `LoopEventType.LOOP_CONTEXT_READ`

读取规则：

- 第 1 次 loop：`LoopContext` 中没有任何 prior loop 信息，读取结果为空摘要。
- 第 2 次及以后：只读取 `loop_index < current_loop_index` 的 observation、error、external reference、state、loop memory 和 event，并压缩成摘要。
- 该摘要作为当前轮 planning 的输入，而不是在接收用户输入时读取。

### 2.7 组装 AgentInput

输入：

- `Goal`
- `List[Memory]`
- 空 `LoopContext`
- `List[ExternalReference]`

输出：

```python
AgentInput
```

字段：

| 字段 | 含义 |
| --- | --- |
| `goal` | 当前目标 |
| `memories` | 短期/长期记忆聚合 |
| `loop_context` | ReAct 执行过程上下文，初始为空，loop 内逐步写入 |
| `external_refs` | skill、MCP、文件、网页、数据库等外部参考 |
| `metadata` | 扩展字段 |

### 2.8 生成当前轮 Thought+Action 协议

输入：

- `AgentInput`
- 已有 `LoopOutput`
- `loop_index`
- 当前 loop 开始时读取的 `LoopContextSummary`

输出：

```json
{
  "thought": "用户输入包含可计算表达式，选择 calculator 工具获取结构化观察结果。",
  "action": {
    "action_type": "tool_call",
    "tool_name": "calculator",
    "arguments": {
      "expression": "100 / 4 + 7"
    },
    "reason": "Numeric expression detected."
  }
}
```

实现：

- `ReactAgentMvp._generate_agent_step_output()`
- `ReactAgentMvp._choose_next_action()`
- `DeepSeekReactPlanner.generate_step_output()`

注意：`thought` 是可展示的推理摘要或计划摘要，不记录隐藏思维链。当前已支持两种 planner：

- 规则版 planner：默认路径，用于本地无网络测试。
- DeepSeek planner：通过 `--planner deepseek` 调用真实 DeepSeek API，让模型输出同样的 Thought+Action 协议。

### 2.9 解析 Thought+Action 协议

输入：

- agent 原始输出文本

输出：

```python
AgentStepOutput(
    thought="...",
    action=Action(...),
    raw_text="..."
)
```

实现：

- `AgentStepOutputParser.parse()`
- `AgentOutputParseError.to_error_feedback()`

解析失败时会生成 `ErrorFeedback`，写入 `LoopContext.errors` 和事件流。该层是输出与输入设计互相对应的协议边界：agent 输出必须能被解析成下一步 loop 输入。

DeepSeek 接入：

| 配置 | 默认值 |
| --- | --- |
| `AGENT_LLM_BASE_URL` | `https://api.deepseek.com` |
| `AGENT_LLM_API_KEY` | 空 |
| `AGENT_LLM_API_KEY_FILE` | `assert/agent_api.txt` |
| `AGENT_LLM_MODEL_NAME` | `deepseek-flash` |
| `AGENT_LLM_TEMPERATURE` | `0.1` |
| `AGENT_LLM_MAX_TOKENS` | `1200` |
| `AGENT_LLM_TIMEOUT_SECONDS` | `90` |

实现：

- `DeepSeekConfig`
- `DeepSeekChatClient`
- `DeepSeekReactPlanner`

### 2.10 判断是否需要工具

输入：

- `user_input`
- 历史 loop 输出

输出：

```python
Action
```

MVP 规则：

| 条件 | Action |
| --- | --- |
| 用户输入包含数学表达式 | `tool_call: calculator` |
| 用户输入包含 “mcp/skill/外部参考/外部工具” | `tool_call: demo_mcp_reference` |
| 用户输入包含“命令/指令/shell/执行” | `command_generation` |
| 其他普通输入 | `tool_call: echo` |
| 已有工具反馈 | `final_response` |

### 2.11 生成 Action

输出：

```python
Action(
    action_type=ActionType.TOOL_CALL,
    tool_name="calculator",
    arguments={"expression": "..."}
)
```

`ActionType` 当前定义：

| 类型 | 含义 |
| --- | --- |
| `tool_call` | 工具调用 |
| `command_generation` | 指令生成 |
| `code_edit` | 代码编辑，预留 |
| `ask_user` | 向用户提问，预留 |
| `memory_write` | 写入记忆，预留 |
| `final_response` | 最终回复 |
| `no_op` | 无动作 |

### 2.12 执行工具 / 指令 / 代码

输入：

- `Action`
- `ToolExecutionLayer`
- `ToolRegistry`

输出：

```python
ToolExecutionResult(
    observation=ToolObservation(...),
    external_reference=ExternalReference(...) | None,
    error=ErrorFeedback(...) | None
)
```

实现：

- `ToolExecutionLayer.execute()`
- `ToolRegistry.execute()`
- `ReactAgentMvp._execute_action()`

当前工具：

| 工具 | 类型 | 输入 | 输出 |
| --- | --- | --- | --- |
| `calculator` | local | `{"expression": "1 + 2"}` | 数值结果 |
| `echo` | local | `{"text": "..."}` | 原文 |
| `command_generator` | local | `{"task": "..."}` | 命令文本草案 |
| `demo_mcp_reference` | mcp | `{"topic": "..."}` | MCP 风格外部参考和使用提示 |

### 2.13 获取 Observation

输出：

```python
ToolObservation(
    tool_name="calculator",
    input={"expression": "(12 + 8) * 3"},
    output=60,
    success=True,
    duration_ms=...
)
```

写入位置：

- `LoopContext.tool_observations`
- `LoopContext.events`
- 成功时可写入 `LoopContext.loop_memories`

### 2.14 写入外部参考信息

当工具类型为 `mcp` 或 `skill` 时，工具返回会被转换成：

```python
ExternalReference(
    ref_type="mcp",
    title="mcp:...",
    content="...",
    source_uri="demo_mcp",
    metadata={
        "usage_hint": "...",
        "loop_index": 1,
        "tool_name": "demo_mcp_reference"
    }
)
```

写入位置：

- `LoopContext.external_refs`
- `LoopContext.events`

实现：

- `ToolExecutionLayer._build_external_reference()`
- `LoopViewUpdater.record_external_reference()`

### 2.15 是否报错 / 记录报错信息

工具失败时输出：

```python
ErrorFeedback(
    source="tool:calculator",
    message="...",
    recoverable=True,
    suggested_fix="Review tool arguments or choose another action."
)
```

写入位置：

- `LoopContext.errors`
- `LoopContext.events`

### 2.16 更新执行状态

输入：

- 当前 loop index
- 执行阶段
- 进度摘要

输出：

```python
ExecutionState(
    status=ExecutionPhase.TOOL_CALLING,
    current_step=1,
    progress_summary="Calling tool: calculator."
)
```

实现：

- `LoopViewUpdater.update_state()`

### 2.17 更新 Loop 短期记忆

输入：

- 工具成功输出

输出：

```python
LoopWorkingMemory(
    title="Tool calculator output",
    content="60",
    loop_index=1,
    expire_policy="current_task"
)
```

实现：

- `LoopViewUpdater.write_loop_memory()`

### 2.18 任务是否完成

MVP 判断：

- 第一次 loop 执行动作拿到 observation。
- 第二次 loop 根据已有 observation 生成 `final_response`。
- 或达到 `max_loops` 后强制总结。

### 2.19 选择输出格式

输入：

- `AgentInput`
- `List[LoopOutput]`

输出：

```python
FinalResponse(format=OutputFormat.MARKDOWN, content="...")
```

MVP 默认 Markdown。后续可根据用户要求切换 `json/table/file_reference`。

### 2.20 生成最终回复

输出：

```python
AgentOutput(
    loop_outputs=[...],
    final_response=FinalResponse(...),
    metadata={"agent_input": ..., "tool_catalog": ...}
)
```

完整结果可写入 JSON 文件，便于调试和后续落库。

## 3. 当前实现和框架图的对应关系

| 框架图概念 | Python 数据类 |
| --- | --- |
| agent 输入 | `AgentInput` |
| 目标/任务 | `Goal` |
| 短期记忆 | `ShortTermMemory` |
| 长期记忆 | `LongTermMemory` |
| loop 短期记忆 | `LoopWorkingMemory` |
| loop 过程信息 | `LoopContext` |
| loop 过程信息摘要 | `LoopContextSummary` |
| 工具调用反馈 | `ToolObservation` |
| 报错 | `ErrorFeedback` |
| 外部参考消息 | `ExternalReference` |
| 执行状态 | `ExecutionState` |
| loop 输出 | `LoopOutput` |
| 行动 | `Action` |
| 最终回复 | `FinalResponse` |
| agent 输出 | `AgentOutput` |

## 4. MVP 边界

当前已完成：

- 数据类定义。
- 内存版记忆存储。
- ReAct loop 事件记录。
- Thought+Action 输出协议。
- Agent 输出解析器。
- DeepSeek planner 接入。
- 工具执行层。
- 工具调用、报错、状态、loop memory 记录。
- MCP 风格外部参考写入。
- 可执行 demo。
- 完整 JSON 输出。

暂未实现：

- 数据库存储。
- MCP/skill 真实接入。
- 代码编辑执行器。
- 用户交互式澄清。
- 长期记忆自动归档策略。

## 5. 后续落库建议

可拆分为以下表：

| 表 | 内容 |
| --- | --- |
| `agent_task` | `Goal` 和任务状态 |
| `agent_memory` | 短期/长期/loop 记忆 |
| `agent_loop_event` | ReAct 事件流 |
| `agent_tool_observation` | 工具调用反馈 |
| `agent_error_feedback` | 报错信息 |
| `agent_external_reference` | 外部参考 |
| `agent_output` | 最终输出 |
