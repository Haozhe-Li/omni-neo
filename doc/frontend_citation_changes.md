# Citation / Source 改动（前端对接说明）

## 1. Source 编号现在是按 thread 累计的，不再每轮清零

以前每次 `/chat` 请求，citation 编号 `[n]` 都从 1 重新数。现在同一个 thread 里，编号是**累计**的：第 1 轮用了 1/2/3，第 2 轮的新 source 会接着从 4 开始，不会重复、不会归零。同一个 url 在整个 thread 里始终复用同一个编号。

对前端的影响：`/chat` 的 SSE 协议（`sources`、`done` 事件）、`GET /api/threads/{thread_id}` 返回的每条消息的 `sources` 字段，**格式完全没变**，还是 `[{n, title, url, content}, ...]`。

**唯一需要注意的行为变化**：模型现在可以直接引用更早轮次已经出现过的 `[n]`，而不用重新搜索/重新读页面。也就是说，**某条消息正文里的 `[n]`，不一定能在这条消息自己的 `sources` 数组里找到**——它可能是前几轮抓到的 source。

→ 建议前端在解析 `[n]` 对应的 source 时，**把整个 thread 里所有消息的 `sources` 数组合并成一张 map（`n -> source`）再查找**，而不是只查当前这条消息自带的 `sources`。每个 source 只会在它第一次被抓取的那条消息里出现一次，之后的消息不会重复带它。

## 2. `POST /check_source`（新接口）

用户勾选一段回答文字，去查这段话具体是从哪个 source 的哪一段原文来的。查询范围是**整个 thread 历史上出现过的所有 source**（不止当前这轮）。

**鉴权**：需要 Bearer Token。thread 权限规则同其他 `/api/threads/{id}` 接口——未被认领的 thread 谁都能查，被别的用户认领了就会 403。

**Request**
```json
{
  "thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "text_selection": "特斯拉 2025 年第四季度营收同比增长 20%"
}
```
- `text_selection` 少于 10 个字符会返回 `{ "error": "Text selection is too short" }`。

**Response**
```json
{
  "match": {
    "n": 3,
    "title": "Tesla Q4 Report",
    "url": "https://...",
    "chunk": "命中的原文片段（网页内容按 ~800 字切块过，google 搜索结果的 snippet 一般就是完整一块）",
    "score": 0.87
  }
}
```
- 没有足够可信的匹配时，`match` 为 `null`。
- 只返回**单条最佳匹配**，不是候选列表。
