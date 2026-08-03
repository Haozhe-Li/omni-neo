# Evaluation 结果接口（前端对接）

给 evaluation 看板用的只读接口。数据由 `python -m evals.cli` 写入,**接口这边没有任何写操作** —— 全是 GET,底层全是 select。

Base URL(本地):`http://localhost:8000`
路由前缀:`/api/evals`

---

## 鉴权

**和产品的其他接口不一样,这里不接受 guest 身份。**

产品里的 `get_current_user` 会把 `X-Guest-Id: guest_<uuid>` 当作合法身份且**不做任何验证** —— 对聊天是合理的(guest 是真实的产品档位,限额在别处管),但对评测数据完全不行:任何人编一个 guest id 就能读到内部的模型跑分、prompt 哈希和成本数据。

所以这些接口只认 Clerk 的 Bearer token:

```
Authorization: Bearer <clerk_jwt>
```

缺失或无效 → `401`。

**再收紧一层(可选)**:设环境变量 `EVAL_ADMIN_USER_IDS`(逗号分隔的 Clerk user id),只有名单内的用户能读,其他人 `403`。不设则任何已登录用户都能读。

---

## 数据模型速览

四层,从粗到细:

```
eval_runs        一次 CLI 调用 × 一个模型     → 总分 / 成本 / prompt 哈希
  └ eval_case_scores   该 run 下每个 case 的多次 repeat 汇总(含 stdev)
      └ eval_results   单次执行(一个 case 的一次 repeat)→ 答案 / 报告 / trace / 指标
          └ eval_checks    单条 rubric 的判定结果 ← **看板的 checklist 读这张**
```

`eval_cases` 是 case 注册表(题目 + rubric 定义),每次跑 CLI 都会从 `cases.yaml` 重新 upsert,前端不用解析 YAML。

---

## 接口

### `GET /api/evals/runs` — run 列表

Query:`model_label` / `label` / `status` / `since`(ISO-8601)/ `limit`(≤500,默认 50)/ `offset`

```json
{ "runs": [ {
  "run_id": "…", "label": "matrix v1", "model_label": "gemma-4-31b-high",
  "provider": "cerebras", "reasoning_effort": "high", "model_family": "gemma-4-31b",
  "status": "done", "score": 0.83, "pass_rate": 0.91,
  "suite_scores": { "web-research": 0.78, "charting": 0.9 },
  "n_cases": 9, "n_errors": 0, "total_cost_usd": 0.42,
  "tool_cache": true, "judge_model": "openai:gpt-5.6-terra",
  "git_sha": "…", "prompt_sha": "…", "skills_sha": "…",
  "started_at": "…", "finished_at": "…"
} ] }
```

`prompt_sha` / `skills_sha` 值得展示:分数变了但两个哈希没变 = 模型或运气的问题;哈希变了 = 你改了 prompt 或 skill。

### `GET /api/evals/runs/{run_id}` — run 详情

一次返回三块(详情页三块都要才能渲染,分三次请求等于三次渲染半屏的机会):

```json
{ "run": {…}, "suites": [ {"suite":"web-research","suite_score":0.78,"n_cases":3,…} ],
  "case_scores": [ {"case_id":"…","suite":"…","score_mean":0.74,"score_stdev":0.03,
                    "pass_rate":0.71,"n_repeats":2,"n_errors":0,
                    "ttft_ms_p50":620,"latency_ms_p50":23400,"cost_usd_mean":0.031} ] }
```

`score_stdev` 是 repeat 存在的意义:同一个模型同一个 case 两次分数差得大,说明是 rubric 或 prompt 不稳定,和"模型变差了"是两回事。

### `GET /api/evals/runs/{run_id}/results` — 单次执行列表

Query:`case_id` / `status`(`ok` / `error` / `timeout`)

**这个接口不返回 `trace` / `final_texts` / `report_md`** —— 深度研究的 case 这三列加起来几百 KB,列表页只是渲染一张分数表,拉全量等于传几十 MB。要正文去下面的详情接口。

返回每行含:分数、`skills_loaded`、`n_tool_calls` / `n_searches` / `n_pages_read`、`n_charts` / `n_maps`、`has_report` / `has_question`、`word_count`、`hit_run_limit`、三个 TTFT、`latency_ms`、`per_turn_latency_ms`、token 四件套、`cost_usd`。

### `GET /api/evals/results/{result_id}` — 单次执行详情

Query:`include_trace`(默认 `true`,设 `false` 可省掉 trace)

```json
{ "result": { …上面所有字段…,
    "final_texts": ["第 0 轮回答", "第 1 轮回答"],
    "report_md": "…", "report_title": "…",
    "trace": [ {"turn":0,"i":1,"name":"read_file",
                "args":{"file_path":"/skills/web-research/SKILL.md"},
                "result_head":"…"} ] },
  "checks": [ { "kind":"deterministic", "key":"skill_loaded:charting",
                "label":"skill loaded (skill=charting)", "turn":null,
                "passed":false, "score":0, "max_score":1, "weight":2,
                "evidence":"never read /skills/charting/SKILL.md; loaded: [...]",
                "reason":null, "detail":{…} } ] }
```

`checks` 就是看板的 rubric checklist。`kind` 区分确定性检查和 judge。**`evidence` 一定要显示** —— 一条飘红但不说哪里错的规则,比不显示还费时间。judge 项的 `evidence` 是模型引用的原文片段,`reason` 是一句话理由。

`final_texts` 是数组,一轮一个元素(trip-advisor / guided-learning 是多轮 case)。

### `GET /api/evals/matrix` — case × 模型矩阵(看板首页)

Query:`label` / `run_ids`(逗号分隔)/ `latest_per_model`(默认 `true`)

```json
{ "models": ["gemma-4-31b-high", "gpt-oss-120b-high", …],
  "cases":  [ {"case_id":"charting/revenue-comparison","suite":"charting"}, … ],
  "cells":  { "charting/revenue-comparison": {
                "gemma-4-31b-high": {"run_id":"…","score_mean":0.9,"score_stdev":0.0,
                                     "pass_rate":1.0,"n_errors":0,"n_repeats":2,
                                     "latency_ms_p50":8100,"cost_usd_mean":0.004} } },
  "runs":   [ {"run_id":"…","model_label":"…","provider":"…","reasoning_effort":"…", …} ] }
```

服务端已经 pivot 好了,前端直接 `cells[case_id][model_label]` 取格子,空格子表示该模型没跑这个 case。

**`latest_per_model` 默认开着是有原因的**:run 是只增不删的,同一个模型的 smoke 跑和正式跑会同时存在,不折叠的话同一个模型会占两列而且 case 覆盖范围还不一样。要看历史对比就传显式的 `run_ids`。

### `GET /api/evals/leaderboard` — 模型排行

Query:`label` / `since`

质量分和延迟、成本并排,一行一个 run。含 `score` / `pass_rate` / `error_rate` / `ttft_ms_p50` / `ttft_answer_ms_p50` / `latency_ms_p50` / `latency_ms_p95` / `turns_mean` / token 均值 / `cost_usd_total` / `cost_usd_per_case`。

两个细节要在 UI 上说清楚:延迟分位**只统计 `status='ok'` 的行**(一个 300s 超时会毁掉整列分位数),但 `error_rate` 是单独一列 —— 一个模型在它没崩的那三分之一 case 上考得好,不叫考得好,这两个数必须挨着显示。

底层视图跨**所有** run 聚合,所以攒了 smoke 跑之后记得传 `label` 或 `since` 过滤。

### `GET /api/evals/family-grid` — provider × effort 网格

Query:`family`(如 `gpt-oss-120b`)

一行一个 `family × provider × reasoning_effort` 格子。适合渲染成二维表:

- **沿行看** = 多花 reasoning token 换到了多少分(配合 `reasoning_tokens_mean`)
- **沿列看** = 换 provider 改变了什么。同权重同 effort,**质量分理应几乎相同**,差异应该只落在 TTFT / latency / 成本上
- 某列质量分差很大 → 大概率是 harness 或某家服务的问题,不是模型能力差异

### `GET /api/evals/check-failures` — rubric 失败率排行

Query:`min_evaluated`(默认 1)/ `limit`

`{key, label, kind, n_evaluated, n_failed, failure_rate}`,按失败率降序。这是"下一步该修什么"的列表。同样跨所有 run 聚合,样本少的项用 `min_evaluated` 过滤掉。

### `GET /api/evals/cases` — case 注册表

Query:`suite`

题目原文(`turns`)、rubric 定义(`rubric`,含每条的 layer / key / args / weight / turn)、`is_negative`、`rubric_version`。用来在跑之前就展示"这个 case 会怎么判",以及 rubric 改版后区分历史数据。

---

## 前端建议

1. **首页放 matrix + leaderboard**,一个看横向对比,一个看质量/延迟/成本三角。
2. **格子点进去 → run 详情 → 单次执行详情**,三层下钻。
3. **checklist 按 `weight` 降序排,失败的排前面**,`weight >= 2` 的是硬性项(计入 `pass_rate`),可以加个标记。
4. **`report_md` 用现成的 markdown 渲染器**,它和产品里 `<report>` 侧栏的内容是同一份,连内嵌的 ```echarts 围栏都一样,可以直接复用 `InlineEcharts`。
5. **trace 渲染成时间线**,`name` = `read_file` 且 `args.file_path` 匹配 `/skills/*/SKILL.md` 的那几步单独高亮 —— 那是 skill 触发的时刻,是整套评测最核心的信号。
