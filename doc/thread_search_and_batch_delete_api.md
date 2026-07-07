# 会话搜索 & 批量删除接口

两个接口都挂在现有的 `/api/threads` 前缀下，鉴权方式与其它 `/api/threads/*` 接口一致：

```
Authorization: Bearer <clerk_token>     # 登录用户
X-Guest-Id: guest_xxx                   # 访客
```

---

## 1. 搜索会话

`GET /api/threads/search`

按标题 + 消息内容模糊搜索当前用户自己的会话，中英文都支持。

**Query 参数**

| 字段 | 必填 | 说明 |
|---|---|---|
| `q` | 是 | 搜索关键词，空字符串直接返回空结果 |
| `limit` | 否 | 返回条数上限，默认 20，最大 50 |

**示例**

```
GET /api/threads/search?q=报销流程&limit=20
```

**响应 200**

```json
{
  "results": [
    {
      "thread_id": "a1b2c3d4-...",
      "title": "关于报销流程的讨论",
      "is_pinned": false,
      "updated_at": "2026-07-06T10:32:00+00:00",
      "snippet": "…出差报销流程一般是先垫付，之后…"
    }
  ]
}
```

- `snippet`：命中关键词附近的一小段原文，纯文本、无高亮标记，前端如需高亮自己按 `q` 匹配加粗。
- 结果按相关度降序，同分按 `updated_at` 降序。
- 未鉴权 → 401；`q` 缺失 → 422。
- 想看某条结果的完整消息，用返回的 `thread_id` 调 `GET /api/threads/{thread_id}`（搜索接口不返回全部消息）。

建议输入时做 ~300ms debounce 再请求。

---

## 2. 批量删除会话

`POST /api/threads/batch-delete`

一次性硬删除多个会话（同时清掉消息记录和后端的对话状态），不可恢复。

**请求体**

```json
{
  "thread_ids": ["thread-id-1", "thread-id-2", "thread-id-3"]
}
```

- 单次最多 100 个，超出部分会被截断忽略。
- 重复 id 会自动去重。

**响应 200**

```json
{
  "deleted": ["thread-id-1", "thread-id-2"],
  "not_found": ["thread-id-3"]
}
```

- `deleted`：实际删除成功的 id（确认归属当前用户）。
- `not_found`：不存在或不属于当前用户的 id，会被跳过，**不会**导致整个请求失败。
- 未鉴权 → 401。
- 建议前端删除前弹二次确认（不可恢复），删除后直接把 `deleted` 里的 id 从本地列表移除即可，不需要重新拉全量列表。
