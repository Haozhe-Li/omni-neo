# Omni Agent — 后端接口文档

> Base URL（本地开发）：`http://localhost:8000`

---

## 鉴权说明

所有标注了 **需要鉴权** 的接口，必须在 Header 中携带以下其中一种身份凭证：

| 场景 | Header | 示例值 |
|---|---|---|
| 已登录用户（Clerk） | `Authorization: Bearer <clerk_jwt>` | `Bearer eyJhbGci...` |
| 游客 | `X-Guest-Id: guest_<uuid>` | `X-Guest-Id: guest_a1b2c3...` |

标注 **可选鉴权** 的接口：不传凭证也能正常调用，但传了之后请求会被绑定到该用户（影响数据归属和保留策略）。

**游客限制：**
- 每日最多 **10 次** AI 请求（`/chat`、`/light_chat`）
- 最多 **5 个** 活跃 Thread（可通过 `GUEST_MAX_THREADS` 环境变量调整）
- Thread 保留 **3 天** 后自动清除

**登录用户：**
- 无以上限制
- Thread 保留 **90 天**，置顶 Thread 永不删除

---

## 目录

- [Thread 管理](#thread-管理)
- [AI 对话](#ai-对话)
  - [/chat 流式响应格式说明](#chat-流式响应格式说明)
- [工具接口](#工具接口)
- [账号与数据](#账号与数据)
- [通用错误码](#通用错误码)
- [前端集成流程参考](#前端集成流程参考)

---

## Thread 管理

### `GET /get_thread_id`

创建一个新 Thread ID。应在每次新建对话时调用。

**鉴权：** 可选（传了会绑定用户）

**Request Headers（可选）**
```
Authorization: Bearer <token>
// 或
X-Guest-Id: guest_<uuid>
```

**Response** — 直接返回 UUID 字符串
```
"550e8400-e29b-41d4-a716-446655440000"
```

**错误**

| 状态码 | 原因 |
|---|---|
| `429` | 游客 Thread 数量已达上限，请登录后继续 |

---

### `GET /api/threads`

获取当前用户的所有 Thread 列表，按 **置顶优先、时间倒序** 排列。

**鉴权：** 必须

**Response**
```json
{
  "threads": [
    {
      "thread_id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "分析苹果公司股票",
      "is_pinned": true,
      "updated_at": "2026-02-23T10:30:00+00:00"
    },
    {
      "thread_id": "661f9511-f30c-52e5-b827-557766551111",
      "title": "帮我写一首诗",
      "is_pinned": false,
      "updated_at": "2026-02-22T08:15:00+00:00"
    }
  ]
}
```

---

### `GET /api/threads/{thread_id}`

获取指定 Thread 中存储的完整消息记录（前端上次 sync 的内容）。

**鉴权：** 必须

**Response**
```json
{
  "messages": [
    { "role": "user", "content": "你好" },
    { "role": "assistant", "content": "你好！有什么可以帮你的？" }
  ]
}
```
> `messages` 的结构与前端 sync 时上传的结构完全一致，后端原样存储原样返回。

---

### `POST /api/threads/{thread_id}/sync`

将前端当前的消息列表同步到云端（**全量覆盖**，非增量）。建议在每次 AI 回复完成后调用。

**鉴权：** 必须

**Request Body**
```json
{
  "messages": [
    { "role": "user", "content": "你好" },
    { "role": "assistant", "content": "你好！有什么可以帮你的？" }
  ],
  "title": "新对话标题"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `messages` | `array` | ✅ | 完整消息列表（全量覆盖） |
| `title` | `string` | ❌ | 同时更新标题，不传则标题不变 |

**Response**
```json
{ "status": "success" }
```

---

### `PATCH /api/threads/{thread_id}/title`

单独修改 Thread 标题（不影响消息内容）。

**鉴权：** 必须

**Request Body**
```json
{ "title": "新标题" }
```

**Response**
```json
{ "status": "updated" }
```

---

### `PATCH /api/threads/{thread_id}/pin`

置顶或取消置顶一个 Thread。置顶 Thread 排列最前，且永远不会被自动清除。

**鉴权：** 必须

**Request Body**
```json
{ "is_pinned": true }
```

**Response**
```json
{ "status": "updated", "is_pinned": true }
```

---

### `DELETE /api/threads/{thread_id}`

永久删除一个 Thread，包括所有对话历史（**不可恢复**）。

**鉴权：** 必须

**Response**
```json
{ "status": "deleted" }
```

| 状态码 | 原因 |
|---|---|
| `404` | Thread 不存在或不属于当前用户 |

---

## AI 对话

### `POST /chat`

主对话接口，支持深度研究、搜索、代码执行等能力。返回 **SSE 流式响应**。

**鉴权：** 可选

**Request Body**
```json
{
  "query": "帮我分析一下特斯拉的最新财报",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "follow_up_content": "上一段内容的文字选区（可选）",
  "personalization": {
    "response_language": "中文",
    "user_local_datetime": "2026-02-23 10:30",
    "user_location": "上海",
    "memories": {
      "user_profile": "金融从业者",
      "current_focus": "新能源行业",
      "interaction_style": "简洁专业",
      "avoid_topics": ""
    }
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `query` | `string` | ✅ | 用户输入 |
| `thread_id` | `string` | ❌ | 对话 Thread ID，不传则无记忆 |
| `follow_up_content` | `string` | ❌ | 用户选中的文字片段（追问场景） |
| `personalization` | `object` | ❌ | 个性化配置，详见下方 |

---

### `/chat` 流式响应格式说明

每条 SSE 事件格式为 `data: <JSON string>\n\n`。所有事件共享如下基础结构：

```json
{
  "type": "...",
  "agent": "...",
  "content": "...",
  "raw": {}
}
```

以下是各 `type` 的详细说明及前端展示要求：

---

#### `type: "answer"` — 最终回答

一次对话中只出现一次，标志流结束。

```json
{
  "type": "answer",
  "agent": "Supervisor",
  "content": [
    { "type": "reasoning", "id": "rs_...", "summary": [] },
    {
      "type": "text",
      "text": "{\"final_answer\": \"# 分析报告\\n...\", \"final_sources\": [{\"title\": \"Tesla Q4\", \"url\": \"https://...\"}]}",
      "annotations": [],
      "id": "msg_..."
    }
  ],
  "raw": {}
}
```

解析步骤：
1. 取 `content` 数组中 `type == "text"` 的那一项
2. 将其 `text` 字段 JSON.parse
3. `final_answer`：Markdown 字符串，渲染展示给用户
4. `final_sources`：来源列表 `[{ "title": string, "url": string }]`，在答案末尾以折叠区块展示

---

#### `type: "reasoning"` — 推理过程

```json
{
  "type": "reasoning",
  "agent": "Sub-agent",
  "content": "需要先搜索近3个月财报数据，然后与去年同期对比...",
  "raw": {}
}
```

展示：默认只显示前 10 个字，其余内容折叠。

---

#### `type: "tool"` — 工具调用

通过 `raw.args` 的内容及 `tool` 字段区分具体工具：

---

##### `tool: "tavily_search"` — 联网搜索

```json
{
  "type": "tool",
  "tool": "tavily_search",
  "agent": "Sub-agent",
  "content": "Tool Calling",
  "raw": {
    "args": { "query": "Tesla Q4 2025 earnings", "max_results": 5, "topic": "general" },
    "id": "fc_..."
  }
}
```

展示：`Searching the web for: "Tesla Q4 2025 earnings"`

---

##### `tool: "skimming_web_pages"` — 网页速读

```json
{
  "type": "tool",
  "tool": "skimming_web_pages",
  "agent": "Sub-agent",
  "content": "Tool Calling",
  "raw": {
    "args": {
      "purpose": "了解特斯拉 Q4 营收情况",
      "urls": ["https://example.com/1", "https://example.com/2"]
    },
    "id": "fc_..."
  }
}
```

展示：
```
Gathering information on: 了解特斯拉 Q4 营收情况
https://example.com/1, https://example.com/2
```

---

##### `tool: "load_web_page"` — 网页精读

```json
{
  "type": "tool",
  "tool": "load_web_page",
  "agent": "Sub-agent",
  "content": "Tool Calling",
  "raw": {
    "args": { "url": "https://example.com/article" },
    "id": "fc_..."
  }
}
```

展示：`Intensive reading: https://example.com/article`

---

##### `tool: "verify_claim"` — 断言验证

```json
{
  "type": "tool",
  "tool": "verify_claim",
  "agent": "Sub-agent",
  "content": "Tool Calling",
  "raw": {
    "args": { "fact": "特斯拉 2025 Q4 营收同比增长 20%" },
    "id": "call_..."
  }
}
```

展示：`Verifying: 特斯拉 2025 Q4 营收同比增长 20%`

---

##### `tool: "write_todos"` — 研究进度 Todo

```json
{
  "type": "tool",
  "tool": "write_todos",
  "agent": "Supervisor",
  "content": "Tool Calling",
  "raw": {
    "args": {
      "todos": [
        { "content": "搜索财报数据", "status": "completed" },
        { "content": "分析同比数据", "status": "in_progress" },
        { "content": "生成报告", "status": "pending" }
      ]
    },
    "id": "call_..."
  }
}
```

展示：在页面右侧固定区域持续更新 Todo 列表（后续事件会覆盖前一次）。

| `status` | 图标 |
|---|---|
| `completed` | ✅ |
| `in_progress` | `⋯`（进行中） |
| `pending` | ○ |

---

##### `tool: "get_navigation"` — 导航/路线规划

```json
{
  "type": "tool",
  "tool": "get_navigation",
  "agent": "Sub-agent",
  "content": "Tool Calling",
  "raw": {
    "args": {
      "origin_lat": 37.7749,
      "origin_lng": -122.4194,
      "destination_lat": 37.8199,
      "destination_lng": -122.4783,
      "mode": "driving"
    },
    "id": "call_..."
  }
}
```

展示：`Getting directions (driving)...`

工具结果会额外附带一条 `type: "navigation"` 事件（与 `weather`/`stock`/`currency` 同机制，从工具返回值里提取），前端可直接用它渲染导航卡片，不需要自己解析 `tool` 事件里的原始输出：

```json
{
  "type": "navigation",
  "agent": "Sub-agent",
  "navigation": {
    "mode": "driving",
    "distance_km": 5.2,
    "duration_min": 14,
    "route_summary": [
      "Head north on Main St (300 m)",
      "Turn right onto Oak Ave (1.2 km)",
      "Arrive at destination"
    ]
  }
}
```

失败时 `navigation.error` 会是一个字符串（例如 OSRM 找不到路线），此时没有 `distance_km`/`duration_min`/`route_summary` 字段，前端应展示一个"暂时无法获取导航信息"之类的兜底文案。

---

##### `tool: "run_python_tool"` — 运行 Python 代码

```json
{
  "type": "tool",
  "tool": "run_python_tool",
  "agent": "Sub-agent",
  "content": "Tool Calling",
  "raw": {
    "args": { "code": "import pandas as pd\nprint(pd.__version__)" },
    "id": "bvstagb3n"
  }
}
```

展示：`Running Python code...`，代码内容默认折叠，展开后显示代码块。

---

#### `type: "error"` — 错误

```json
{
  "type": "error",
  "agent": "system",
  "content": "Stream ended. Answer might escaped."
}
```

展示：提示用户"服务异常，请重试"。

---

### `POST /light_chat`

轻量对话接口，**非流式**，适合简单问答（速度更快）。

**鉴权：** 可选

**Request Body**：与 `/chat` 相同

**Response**
```json
{
  "answer": "你好！有什么可以帮你的？",
  "use_search": false
}
```

| 字段 | 说明 |
|---|---|
| `answer` | Markdown 格式的回答 |
| `use_search` | 是否调用了联网搜索 |

---

### `POST /research_helper`

判断是否适合展开深度研究，并对 query 进行改写优化。用于在发起 `/chat` 前的预判断。

**鉴权：** 可选

**Request Body**：与 `/chat` 相同

**Response**
```json
{
  "response": "好的，我来帮你深入研究特斯拉财报...",
  "read_to_begin_research": true,
  "rewritten_query": "Tesla 2025 Q4 earnings report financial analysis"
}
```

| 字段 | 说明 |
|---|---|
| `response` | 给用户看的简短回复 |
| `read_to_begin_research` | `true` 时前端可展示"开始研究"按钮或直接发起 `/chat` |
| `rewritten_query` | 优化后的查询词，可替换发给 `/chat` 的 `query` |

---

## 工具接口

### `POST /api/sst`

语音转文字（SST/STT）接口。前端上传音频文件，后端返回识别文本。

**鉴权：** 无需（匿名可调用，不计入游客额度）

**Request Headers**
```
Content-Type: multipart/form-data
```

**Form Data**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | `binary` | ✅ | 音频文件（`audio/*`） |

**Response**
```json
{ "text": "你好，这是语音识别结果。" }
```

| 状态码 | 原因 |
|---|---|
| `400` | 不是音频文件，或文件为空 |
| `500` | 服务端识别失败（例如 `GROQ_API_KEY` 未配置） |

---

### `POST /get_title`

根据第一条消息自动生成 Thread 标题。建议在用户第一次发送消息后异步调用。

**鉴权：** 无需

**Request Body**
```json
{ "query": "帮我分析一下特斯拉的最新财报" }
```

**Response** — 直接返回字符串
```
"特斯拉财报分析"
```

---

### `POST /get_model`

根据用户输入推荐使用 `canvas`（深度）还是 `light`（轻量）模式。

**鉴权：** 无需

**Request Body**
```json
{ "query": "帮我写一首诗" }
```

**Response**
```json
{ "model": "light" }
```

| `model` | 推荐接口 |
|---|---|
| `canvas` | `/chat` |
| `light` | `/light_chat` |

---

### `POST /check_source`

验证一段文字与给定来源的相关性。

**鉴权：** 无需

**Request Body**
```json
{
  "source": {
    "sources": [
      { "title": "Tesla Q4 Report", "url": "https://...", "content": "..." }
    ]
  },
  "text_selection": "特斯拉 2025 年第四季度营收同比增长 20%"
}
```

> `text_selection` 长度必须 ≥ 10 个字符，否则返回 `{ "error": "Text selection is too short" }`。

---

### `POST /update_memories`

基于历史对话，用 LLM 更新用户的记忆摘要，供下次对话 `personalization.memories` 使用。

**鉴权：** 无需

**Request Body**
```json
{
  "past_queries": ["帮我分析特斯拉", "写一首关于秋天的诗"],
  "past_memories": {
    "user_profile": "金融从业者",
    "current_focus": "新能源",
    "interaction_style": "简洁",
    "avoid_topics": ""
  }
}
```

**Response** — 更新后的 `Memories` 对象
```json
{
  "user_profile": "金融从业者，关注新能源行业",
  "current_focus": "新能源、电动车",
  "interaction_style": "简洁专业",
  "avoid_topics": ""
}
```

---

### `GET /health`

健康检查。同时在后台触发过期 Thread 的清理任务。

**鉴权：** 无需

**Response**
```json
{ "status": "ok" }
```

---

## 账号与数据

### `POST /api/users/merge`

游客登录后，将游客期间产生的所有 Thread 迁移到正式账号。  
**应在用户完成 Clerk 登录后立即调用一次。**

**鉴权：** 必须（Bearer Token）

**Request Body**
```json
{ "guest_id": "guest_a1b2c3d4-..." }
```

**Response**
```json
{ "status": "merged", "threads_migrated": 3 }
```

| 状态码 | 原因 |
|---|---|
| `400` | `guest_id` 格式不合法 |
| `403` | 当前身份仍是游客，无法合并 |

---

## 通用错误码

| 状态码 | 含义 | 前端处理建议 |
|---|---|---|
| `401` | 未提供合法身份凭证 | 引导用户登录 |
| `403` | 无权操作该资源 | 提示"权限不足" |
| `404` | 资源不存在或不属于当前用户 | 提示"不存在或已删除" |
| `429` | 达到使用上限（游客每日请求 / Thread 数量） | 引导用户登录 |
| `500` | 服务器内部错误 | 提示"服务异常，请稍后重试" |

---

## 前端集成流程参考

### 新建对话

```
1. GET  /get_thread_id             （携带身份 Header）→ 拿到 thread_id
2. POST /get_model    { query }    → 决定用 /chat 还是 /light_chat
3. POST /chat 或 /light_chat       → 展示回答
4. POST /get_title    { query }    → 异步更新侧边栏标题
5. POST /api/threads/{id}/sync     → 持久化消息 + 标题到云端
```

### 切换到历史对话

```
1. GET /api/threads                          → 获取 Thread 列表，渲染侧边栏
2. 用户点击某条 Thread
3. GET /api/threads/{thread_id}              → 恢复消息记录，渲染到聊天区
```

### 游客登录转正

```
1. 用户完成 Clerk 登录
2. 读取 localStorage 中的 guest_id
3. 如果存在 guest_id：
   POST /api/users/merge  { "guest_id": "guest_xxx" }  （携带 Bearer Token）
   → 成功后 localStorage.removeItem('guest_id')
```
