# Omni Evaluation — 设计方案

针对 **pro 模式**、按 **skill** 组织的行为评测。目标是回答两个问题:

1. **触发正确吗** — 给定一个 query,agent 有没有识别出该用哪个 skill、有没有真的去读它、有没有在不该用的时候乱用。
2. **执行到位吗** — 读了 skill 之后,有没有遵循 skill 里写的流程,最终产物(报告/图表/地图/问题块)是否符合 skill 定义的契约,以及内容质量如何。

结果全部落 Supabase,前端另做渲染。

---

## 1. 为什么不走 `/chat` HTTP 端点

评测直接在进程内构建 agent,不经过 FastAPI。理由:

- `/chat` 需要 Clerk JWT、配额扣费(`user_usage`)、Redis checkpointer、thread 记账、widget predictor —— 这些都和"agent 行为好不好"无关,但每一个都可能让评测跑失败或污染生产数据。
- 进程内可以拿到**完整 message 列表**(每个 AIMessage 的 tool_calls + 每个 ToolMessage 的结果),这是所有确定性打分的原料。SSE 流是有损的。
- 可以自由替换模型、关掉 fallback。

代价:不覆盖 SSE 协议层(`_normalize_citations`、`drafting` 事件、artifact 拦截)。这部分归单元测试,不归本评测。**唯一例外**:打分前对最终文本跑一遍 `core.stream._normalize_citations`,因为生产环境确实会修复引用格式,不该重复扣分。

### Agent 构建(`evals/agent_factory.py`)

复刻 `core/stream.py::_stream_agent` 的输入构造,但三处改动:

| 改动 | 原因 |
|---|---|
| `checkpointer` 换成 `InMemorySaver` | 不污染 Upstash,每个 case 独立 thread |
| **移除 `ModelFallbackMiddleware`** | 否则 Cerebras 挂掉时静默 fallback 到 Gemini,你以为在测 gemma 实际在测 gemini —— 整批数据作废且无感知 |
| `model` 参数化 | 模型矩阵的前提,`build_agent("pro")` 目前硬编码 `pro_llm` |

其余保持一致:`PRO_PROMPT`、`RETRIEVAL_TOOLS`、`ToolCallLimitMiddleware(run_limit=30)`、`input_state["files"] = PRO_SKILL_FILES`。

`<personalization>` 块固定注入(语言 / 地点 / **写死的日期时间**),否则"今年"、"最近"类问题跨天不可复现。

调用用 `agent.astream(stream_mode=["messages", "updates"], subgraphs=True)` —— 和 `core/stream.py` 完全一样的调用方式。**不能用 `ainvoke`**:TTFT 必须靠流式逐 chunk 打时间戳才测得到。两个 stream_mode 各司其职:

- `messages` → 逐 chunk 时间戳,算 TTFT
- `updates` → 完整的 AIMessage / ToolMessage,算 trace、turns、token 用量

---

## 2. 一个 case 长什么样 —— 全部配置在 `evals/cases.yaml`

**代码里不出现任何 query、阈值、权重、judge 提问。** Python 侧只实现 check 的**类型**(怎么判),
所有参数在 `evals/cases.yaml` 里。改评测标准 = 改 YAML,不用动代码、不用发版。

这不只是洁癖:字数下限这类东西每个 query 天然不同(闲聊 100 字上限 vs deep research 800 字下限),
搜索次数区间也是,写死就等于把"什么算好"焊死在代码里。而且 rubric 是要反复调的 —— 调它的人不该
需要读 Python。

```yaml
- id: web-research/sea-lions-vs-seals
  suite: web-research
  skill: web-research
  title: 深度研究海狮与海豹的区别
  turns:
    - text: 深度研究一下海狮和海豹的区别

  # 层 A:确定性检查,Python 判定,0 方差
  checks:
    - {key: skill_loaded,     args: {skill: web-research}, weight: 3}
    - {key: tool_called,      args: {tool: google_search, min: 3, max: 12}, weight: 2}
    - {key: distinct_queries, args: {tool: google_search, min: 3}, weight: 2}
    - {key: has_report,       args: {min_words: 800, require_title: true}, weight: 3}
    - {key: chart_count,      args: {min: 2}, weight: 3}
    - {key: citation_count,   args: {min: 6}, weight: 2}

  # 层 B:LLM judge,只判确定性检查判不了的
  judge:
    - key: answers_question
      weight: 3
      prompt: 报告是否直接、完整地回答了「海狮和海豹的区别」?是否覆盖了外耳、前肢/后肢与运动方式…
```

YAML 顶部有 `defaults:` 块:`personalization`(含写死的日期时间)、`repeats`、`timeout_s`,以及
`common_checks` —— 格式合规那一组(tool_discipline / no_hyperlinks / citation_exists …)自动合到
每个 case 上,不用逐个抄。case 里写 `disable: [citation_exists]` 可以单独关掉某项(写作类和闲聊
case 就不该要求引用)。

`weight >= 2` 的 check 算**硬性项**,计入 `pass_rate`。`turn` 缺省 `last`,多轮 case 才需要显式写。

启动时把整个 YAML upsert 进 `eval_cases` 表,前端不解析代码也能渲染 rubric。

### 多轮

`trip-advisor` Step 0 **强制**先发 `<question>` 然后停;`guided-learning`、`web-research` 也可能问。所以 `turns` 是列表 —— 第 k 轮跑完把第 k+1 条脚本消息喂进同一个 thread。用**写死的脚本回复**而不是 LLM 模拟用户,保证可复现。agent 若没问问题,脚本回复照喂,不影响。

每轮可以有自己的 `checks`(`turn=0` / `turn=1` / `turn="last"` / `turn="any"`)。比如 trip-advisor:

- turn 0 必须有 `QuestionBlock()` 且 `NoReport()`(过早写报告 = 没遵循 Step 0)
- turn 1 必须有 `HasReport()` + `MapFence()` + 天气工具调用

---

## 3. 层 A:确定性检查库(`evals/checks.py`)

用户列的 rubric 里绝大部分属于这层。零方差、零成本、可直接给前端当 checklist 渲染。

检查的**类型**在这里实现,**参数**全在 `cases.yaml`。29 个类型,分五组。

### 3.1 Skill 触发(3 个)

| key | 判定方式 |
|---|---|
| `skill_loaded` | trace 里存在 `read_file(file_path="/skills/<skill>/SKILL.md")` |
| `skill_not_loaded` | 反向,用于负例 |
| `skill_load_order` | before 的读取早于 after(web-research 要求 report 阶段才加载 charting) |

之所以完全确定性,是因为 deepagents 的 `SkillsMiddleware` 就是让模型用 `read_file` 去读 SKILL.md —— 这在 trace 里是一次实打实的工具调用,不需要猜。

### 3.2 工具使用(5 个)

| key | 判定方式 |
|---|---|
| `tool_called` | 某工具调用次数落在 `[min, max]` |
| `no_tool_calls` | 一次都没调(闲聊) |
| `distinct_queries` | **去重后**的不同查询数 —— 防"同一个 query 搜三遍"凑数 |
| `distinct_domains` | `load_web_page` 命中的不同域名数 —— 防"六个来源全是同一个站" |
| `search_discipline` | 同一 sub-topic 的 `google_search` ≤ N(PRO_PROMPT 里的硬限制) |

`distinct_*` 这两个是有意加的:光看调用次数,一个模型把同一个 query 搜五遍也能拿满分,但那是刷指标不是做研究。

### 3.3 产物契约(9 个)

| key | 判定方式 |
|---|---|
| `has_report` / `no_report` | 正则抓 `<report title="…">…</report>`,校验 title 属性存在、正文字数达标 |
| `chart_count` | ` ```echarts ` 围栏个数落在区间 |
| `charts_valid` | 每个围栏 `json.loads` 通过、有非空 `series`;`require_palette` 时校验颜色取自 charting skill 那 6 色 |
| `map_fence` | ` ```map ` 围栏 JSON 合法、有 `title`、`pins ≤ 8`、每个 pin 有 `name` |
| `no_map` | 反向 |
| `question_block` | `<question>` JSON 合法、`id` 唯一、`type ∈ {single,multiple,text}`、**且是整条消息最后一个 block** |
| `no_question_block` | 反向 |
| `has_delimiters` | 写作类:一句说明 + `---` 包裹正文 |
| `followup_question` | 写作类:结尾一个追问 |

这一组全是**契约校验**:前端靠正则/JSON 解析这些块来渲染,格式坏掉 = 功能坏掉,和内容好不好无关。所以必须确定性判,不能交给 judge。

### 3.4 格式合规(11 个)

全部一一对应 `core/agent.py` 里 `PRO_PROMPT` 的硬约束 —— 这层本质上是在问"prompt 里写的规矩,模型到底听没听"。

| key | 判定方式 |
|---|---|
| `word_count` | **CJK 感知**:CJK 字符数 + 拉丁词数相加 |
| `citation_count` | 不同 `[n]` 的个数 ≥ min |
| `citation_exists` | 每个 `[n]` 都在本轮 registry 里 —— **抓幻觉引用** |
| `citation_format` | ASCII 括号、成簇在段末、不在句中 |
| `citation_coverage` | 含工具事实的段落里带引用的比例 |
| `no_hyperlinks` | 无 `[text](url)`、无裸 URL |
| `no_ascii_art` | 代码围栏内若只有 `│├└+-\|` 类字符即失败 |
| `tool_discipline` | 不存在同时含正文和 `tool_calls` 的 AIMessage |
| `no_leading_header` | 首行不是 `#` |
| `latex_sanity` | 用 `\( \)` / `\[ \]` 不用 `$`;普通数字/日期/单位不包 LaTeX |
| `no_prompt_leak` | 复用 `core.prompt_guard.has_prompt_leakage`,不另写一套 |
| `response_language` | 回答语言是否等于期望语言(见 3.7) |

`no_prompt_leak` 有个坑值得写下来:`core/prompt_guard.py` 的 guard 初始化成**空指纹集**,只有
`main.py` 启动时调 `register_sensitive_prompts` 才装弹,而评测从不导入 `main.py`。不装弹的话
`has_prompt_leakage` 对任何输入都返回 False —— 这个 weight=3、施加在全部 case 上的检查会恒定通过,
凭空抬高每个分数。所以 `cli.py` 启动时显式装弹,**并且**这个 check 自己会检测 guard 是否为空,
空则直接判失败。一个没法干活的检查必须大声失败,而不是把所有人放行。

### 3.5 运行指标 —— 见第 4 节,单独一层,记录不打分。

---

### 3.6 citation 到底验到什么程度

单独说,因为这是最容易只做表面文章的一项。分四级,前三级确定性,第四级 judge:

**L1 格式** (`citation_format`,确定性) —— ASCII `[1]` 而非全角 `【1†L1-L3】`,成簇出现在段末而非句中散落。注意打分前会先跑一遍 `core/stream.py` 的 `_normalize_citations`,因为生产环境确实会修复一部分格式,不该重复扣分 —— 验的是**归一化之后仍然错**的情况。

**L2 存在性** (`citation_exists`,确定性,**最有价值的一项**) —— 答案里每一个 `[n]`,都必须在这一轮 `all_citations()` 注册表里真实存在。registry 是**工具自己**在返回结果前调 `register_citation` 建立的,所以它就是"模型合法可引用的全集"。模型写了 `[7]` 但这轮只注册到 `[5]`,就是凭空捏造引用编号 —— 这是个真实且隐蔽的 bug 类型:前端渲染出一个点不开或指向错误来源的角标,用户看到的是"有出处",实际没有。这一项 weight 给 3。

**L3 覆盖率** (`citation_coverage`,确定性但是启发式) —— 反过来查漏引:段落里出现了数字、年份、专有名词等明显来自检索的内容,却整段没有 `[n]`。启发式必然有假阳性,所以阈值是比例(`min_ratio: 0.6`)而不是全有全无,权重也压低。

**L4 溯源(已移除)** —— 曾经有一级 judge 把 `[n]` 对应的完整 ToolMessage 原文和被引用的句子一起送去问"这段原文是否支持这个说法",由 `cases.yaml` 的 `citation_grounding` 开关控制。

它在 6 个强模型上的通过率是 **0/15**,阈值从 0.8 降到 0.6 之后依然全灭 —— 也就是说它从来没有把任何两个模型区分开,只是给每个 web-research / report-writing case 加了一个 weight 3 的固定扣分。根因是 claim 粒度:一个 claim 是一整行,常常捆着 2-3 个子事实,只要一个子事实没被来源直接覆盖(典型的是"分类学正式名称"这类补充说明),整行就判 0。

要重新引入的话,得先把 claim 切到子句级别再判,而不是调阈值 —— 阈值已经调过一次,没用。眼下抓伪造引用的职责由 L2 `citation_exists` 承担,它比对的是工具真实注册过的编号,0 方差。

---

### 3.7 语言一致性

单独一个维度:**回答语言跟不跟得上**。`response_language` 进了 `common_checks`,所以 28 个 case
全都在测这一项,另有 4 个 case 专门针对它。

判定用**字符类占比**而不是语言识别模型:这里只需要分中英,信号在码点层面就是确定的,而确定性检查
不该依赖模型下载或网络调用。CJK 按字符计、英文按词计,`ratio = cjk / (cjk + latin)`。

两条阈值(≥0.55 判中文、≤0.15 判英文)拉得很开,因为两种语言都会合法地借用对方:解释「你好」
是什么意思的英文回答必然引用中文字符,讲 LangChain 估值的中文回答满是英文产品名和数字。落在中间
带的判 `mixed`,给 0.5 分而不是 0 —— 真正半中半英的回答是另一种失败,和答错语言不是一回事。
实测:中文技术回答 0.71–0.73,英文解释中文词 0.14,真正半半的 0.45。

**期望语言从哪来**,完全照production的规则:

- `<personalization>` 里写了 `Response language` → 它赢。PRO_PROMPT 明确说「Honour it silently」。
- 没写 → 跟随 query 的语言。

第二条要能测,`build_personalization` 必须在 case 没指定语言时**整行省略**,而不是填个默认值 ——
填了就等于替模型把题答了,检查永远通过。所以 language suite 的 case 写 `personalization: {language: null}`。

混语 query 是重点,因为判据是**句子骨架**属于哪种语言,不是里面出现了什么字符:

| query | 期望 | 为什么 |
|---|---|---|
| 你可以告诉我the current valuation of langchain吗? | zh | 中文句式,英文只是术语 |
| could you please explain to me what does 你好 mean? | en | 英文句式,中文只是被讨论的对象 |

模型最常见的错误就是被**关键词的语言**带跑,用户感受到的是助手无缘无故中途换语言。所以
`expect_lang` 写在 case 里而不是自动检测 query —— 这几个例子恰恰是为了打败朴素检测才挑的。

还有一个 case 测 personalization 压过 query 语言(英文 query + 中文 personalization → 必须答中文),
外加一条 judge `silent_compliance`:换语言时不能自我解释或道歉。

---

## 4. 层 C:运行指标(`evals/metrics.py`)

不打分,单独一层。质量分决定"哪个模型答得好",这层决定"哪个模型用得起、等得起" —— 对 pro 模式尤其关键,因为一次 deep research 是 20+ 次 LLM 往返。

### 延迟

agent 场景下 TTFT 不是一个数,是三个,而且差着一个数量级:

| 指标 | 定义 | 意义 |
|---|---|---|
| `ttft_ms` | agent 吐出**任何** chunk 的时刻(通常是 reasoning token 或第一个 tool_call chunk) | 用户感知的"它动了"。对应前端第一个 `reasoning`/`tool_call` SSE 事件 |
| `ttft_answer_ms` | **最终答案正文**第一个 token(所有工具调用之后) | 用户感知的"它开始回答了"。deep research 下可能 60–120s |
| `ttft_report_ms` | 文本流里出现 `<report` 的时刻 | 驱动前端侧栏 "Writing report…" 展开 |
| `latency_ms` | 从请求发出到 stream 结束 | 总耗时 |

多轮 case 每轮各记一组,另存 `per_turn_latency_ms` 数组。

### 轮次

`n_llm_turns` = 一次 run 里 AIMessage 的个数,即 agent loop 的 LLM 往返次数。同时记 `hit_run_limit`(是否被 `ToolCallLimitMiddleware(run_limit=30)` 截断)—— 触顶意味着答案是残缺的,任何质量分都要连带打折。

### Token

**实测结论(反直觉,必须按这个来实现)**:三家 provider 的流式 usage 语义不一样。

- **Cerebras / Groq** —— usage 只在**最后一个 chunk** 一次性给出**总量**。
- **Gemini** —— usage 是**逐 chunk 增量**:`input_tokens` 只在首 chunk 出现,`output_tokens` 分散在各 chunk,而**最后一个 chunk 是全 0**。

所以"取最后一个非空 `usage_metadata`"这种写法在 Gemini 上会读出 0。唯一通用的做法是**跨 chunk 累加**(LangChain `AIMessageChunk.__add__` 就是这个语义),两种约定下都正确。实现上直接从 `updates` 流拿聚合好的完整 AIMessage —— LangGraph 已经用 `+` 合并过了 —— 再对一次 run 里所有 AIMessage 求和。

记录字段:

- `input_tokens` / `output_tokens` —— **对所有 LLM 调用求和**,这是计费口径
- `cached_input_tokens` —— 来自 `input_token_details.cache_read`,Cerebras 和 Gemini 都给,成本要按折扣价算
- `reasoning_tokens` —— 来自 `output_token_details.reasoning`,gpt-oss / gemini-3-flash 都给。**这是 reasoning_effort 三档对比的核心指标**
- `peak_context_tokens` —— 单次 LLM 调用的最大 input,即上下文峰值

注意 agent 的 token 结构:每一轮都要重发整个对话历史,所以 `input_tokens` 总和大致随轮次**平方增长**。一次 25 轮的 deep research,input 通常是 output 的 10 倍以上 —— 成本几乎全在 input 上,`cache_read` 命中率因此比模型单价更能决定实际花费。

### 成本

**价格不写在代码里**,单独一张 `eval_pricing` 表(model_id + effective_from + 三档单价)。理由:价格会变,而历史 run 的 token 数是不变的 —— 把价格放表里,成本随时能在 SQL 里重算,不用重跑评测。`eval_runs.pricing_version` 记下算分时用的是哪一版。

没有录入价格的模型,`cost_usd` 写 NULL 而不是 0 —— 宁可显示"未知"也不显示一个错的便宜价。

## 5. 层 B:LLM judge(`evals/judge.py`)

只判断层 A 判不了的:**答没答、答得好不好、流程像不像**。判据 prompt 全部逐 case 写在 `cases.yaml` 里。

**判定原则**:凡是能确定性判的,一律不给 judge。judge 有方差、有成本、会被长文本讨好 —— 让它去数图表个数或校验 JSON 是在浪费它,也是在给分数注入噪声。它只负责"读懂内容才能回答"的问题。

- judge 模型 **`openai:gpt-5.6-terra`**,刻意选在被测集合之外:`core/llm.py` 里没有任何候选和它同源同厂,所以不存在模型自评、也不存在同一家 provider 的兄弟模型互评。
- **judge 的 rubric 全部用英文写**,即便被测 query 是中文。判据文本是给 judge 读的,也是给排查分数的人读的,统一一种语言少一层歧义。
- 输入:user query + 最终答案全文 + report 正文 + **压缩后的 trace**(工具名 + 参数摘要 + 结果前 200 字的有序列表)。
- 输出:结构化(Pydantic + `with_structured_output`),每条 rubric 给 `score ∈ {0,1,2}` + `evidence`(必须引用原文片段)+ `reason`。强制 evidence 是为了压住 judge 瞎给分。
- 默认跑 1 次;`--judge-repeats 3` 取中位数,用于测 judge 自身方差。

### 当前用到的 judge 维度(42 条,跨 28 个 case)

按性质分五族:

**A. 回答质量** —— 最基本的那个问题
`answers_question` 有没有真的回答 / `substance` 是具体事实还是空话 / `structure` 结构是否清晰 / `recency` 是否用了当期信息 / `disagreement` 来源冲突时是否如实呈现

**B. 流程遵循** —— skill 里写的工作流,有没有照做
`research_arc` 是否 orient→dive→compare 而非平铺搜索 / `calibration` 教学是否针对用户自报的水平 / `concept_order` 概念是否按依赖顺序拆解 / `has_interaction` 有无检验理解的环节 / `is_quiz` 出题模式是否真的在出题 / `asks_right_thing` 澄清问题是否问在点上 / `question_quality` `options_sane` 问题与选项质量

**C. 事实性 / 溯源** —— 抓幻觉,这一族最重要
`factual` 是否与 SKILL.md 一致(about 类)/ `number_from_tool` 数字是否与工具返回一致 / `numeric_correct` 计算结果是否正确 / `no_mental_math` 数值是否真来自 `run_python` 而非心算后补个代码块 / `chart_data_real` 图表数据是否来自检索

`places_real`(推荐的店是否来自 places 工具)和 L4 `citation_grounding` 都已移除,原因见 §3.6 和 `cases.yaml` 里 mapping case 上的注释 —— 两者在所有强模型上的通过率都是 0,是恒定扣分而非区分度。`places_real` 那条尤其是 case 自身的问题:query 要五家店,而 `google_search_places` 的 k 上限就是 5、实测只返回两三家,模型只能少给或被判 0。

**D. 约束遵守** —— 用户明确说了的,有没有落实
`honors_constraints` 行程是否落实每一项硬约束 / `obeys_constraint` "用一句话"是否真的一句话 / `right_sized` 篇幅是否与问题匹配 / `scaled_to_trip` 两日行程是否按比例缩减而非套七日模板 / `itinerary_feasible` 地理时间上是否可行 / `budget_realistic` 预算拆分是否合理 / `separation` `tone` 写作类的格式与语气

**E. 安全 / 语言** —— `treats_as_data` 网页里的伪指令是否被当数据而非指令 / `refuses_gracefully` 拒绝时是否仍然友好有用 / `silent_compliance` 换语言时是否不做自我解释

C 族里几条(`number_from_tool`、`chart_data_real`、`no_mental_math`)判的都是同一件事:**模型写出来的数字/名字,是不是它工具里真拿到的那个**。这是 pro 模式最贵的失败 —— 答案看起来有理有据、有图有引用,但数字是编的。这类必须 judge,因为要把答案里的数和 trace 里的数对着看;但正因为判据明确(对得上/对不上),judge 在这上面反而比在"质量好不好"上稳定得多。

---

## 6. Case 清单(28 个)

**完整定义在 `evals/cases.yaml`,这里只列分布。**清单不在本文档里重复一遍 —— 两处写同一件事,
迟早会有一处过期,而过期的那处一定是文档。

| suite | 数量 | 其中负例 |
|---|---|---|
| web-research | 3 | 0(含 1 个「应先澄清」) |
| report-writing | 2 | 1 |
| charting | 2 | 1 |
| mapping | 2 | 1 |
| ask-question | 2 | 1 |
| guided-learning | 2 | 0(1 个多轮) |
| trip-advisor | 2 | 0(均多轮) |
| about | 2 | 0 |
| general | 5 | 1 |
| language | 4 | 0 |
| adversarial | 2 | 0 |
| **合计** | **28** | **5** |

合计约 320 项确定性检查 + 42 项 judge。

每个 suite 都配**负例**,因为"不该触发时不触发"和"该触发时触发"同等重要 —— pro 模式最典型的失败
模式是用力过猛(简单问题写 800 字报告 + 两张图)。三个多轮 case(guided-learning ×1、trip-advisor ×2)
覆盖"先提问再产出"的流程,其中 trip-advisor 的 Step 0 是硬门槛:第一轮直接出行程即判失败。

---

## 7. 跨模型可比性 —— 工具缓存

`google_search` 打的是真实 API。同一个 case 在模型 A 和模型 B 上跑,搜到的网页可能不同,**分数差异会被搜索运气污染**。

方案:`--tool-cache` 模式,包一层 memoize,按 `(tool_name, canonical_args)` 落盘 JSON。

- **跨模型对比 → 缓存开**。所有模型看到同一份证据,差异纯粹来自模型行为。
- **回归/线上健康度 → 缓存关**。测真实链路。

建议默认开,这是让模型矩阵结论站得住脚的关键一步。

---

## 8. 模型矩阵

`core/llm.py` 里全部 **13 个 chat 模型**都测,只排除 `prompt_guard_2_86m`(Prompt Guard 2 是 86M 的注入分类器,不是对话模型,喂给它 pro prompt 无意义)。

模型清单从 `core/llm.py` **反射式读取**而不是在 evals 里抄一份:遍历模块里所有 `BaseChatModel` 实例,减去一个显式的 `_NOT_CHAT_MODELS = {"prompt_guard_2_86m"}` 排除集。这样 llm.py 里新加一个模型,评测自动覆盖,不会悄悄漏测。

| label | 变量 | 分组意义 |
|---|---|---|
| `gemma-4-31b-high` | `gemma_4_31b_high` | **当前 pro 生产模型,基线** |
| `gemma-4-31b-low` | `gemma_4_31b` | 同模型低 effort,量化 effort 的收益 |
| `gpt-oss-120b-low` | `gpt_oss_120b_low` | **gpt-oss-120b 2×3 组**,详见下方。low/cerebras 是当前 fast 生产模型 |
| `gpt-oss-120b-medium` | `gpt_oss_120b_medium` | |
| `gpt-oss-120b-high` | `gpt_oss_120b_high` | |
| `gpt-oss-120b-low-groq` | `gpt_oss_120b_low_groq` | |
| `gpt-oss-120b-medium-groq` | `gpt_oss_120b_medium_groq` | |
| `gpt-oss-120b-high-groq` | `gpt_oss_120b_high_groq` | |
| `gpt-oss-20b` | `gpt_oss_20b` | 小模型下限 |
| `qwen3.6-27b` | `qwen_3_6_27b` | |
| `gemini-3-flash` | `gemini_flash` | |
| `gemini-flash-lite` | `gemini_flash_lite_latest` | 当前 pro fallback + scheduled 模型 |
| `llama-3.1-8b` | `llama3_1_8b` | 小模型下限 |

### gpt-oss-120b:完全交叉的 2×3

六个 variant 不是六个独立模型,是 **provider × reasoning_effort 的全交叉设计**,单独当一组分析:

| | low | medium | high |
|---|---|---|---|
| **Cerebras** | `gpt-oss-120b-low`(fast 生产) | `gpt-oss-120b-medium` | `gpt-oss-120b-high` |
| **Groq** | `gpt-oss-120b-low-groq`(fast fallback) | `gpt-oss-120b-medium-groq` | `gpt-oss-120b-high-groq` |

全交叉才能把两个因素**拆开**归因,这是它比"随便测几个"强的地方:

- **沿行看** = effort 的边际收益。配合 `reasoning_tokens`,直接算出"每多烧 1 万 reasoning token 换到多少质量分",而且能在两个 provider 上各验一遍 —— 如果两条曲线形状一致,这个结论就是模型属性而不是某家服务的偶然。
- **沿列看** = provider 效应。**同权重同 effort,质量分理应几乎相同**,差异应该全部落在 TTFT / latency / 单价上。
- **交互项** —— 如果某个格子明显偏离,那说明有个只在特定 provider+effort 组合下才出现的问题。

第二点其实是这套评测的**内建对照组**:同一个模型、同样的权重、同样的 prompt,质量分若出现大的系统性差距,更可能是 harness 或某家服务的问题(量化差异、默认 max tokens 不同、tool-call 格式处理不同、reasoning 泄漏进 content),而不是真实的模型能力差异。所以这三对是**评测自身效度的 sanity check** —— 它们分数对不上,先别急着下模型结论,先查框架。

已知的一处非对称,实现时要留意别误判成 provider 效应:Groq 那三个显式设了 `reasoning_format="parsed"`,Cerebras 那三个靠 `core/llm.py` 里 `ChatCerebras` 子类覆写 `_convert_chunk_to_generation_chunk` 来抽 reasoning。两条路径都是把 reasoning 挡在 `content` 外面,目的一致,但实现完全不同 —— 如果只在 Cerebras 侧看到 `<think>` 漏进正文、或者 `reasoning_tokens` 一侧恒为 0,那是这个补丁的问题,不是模型的。

### 其余两组对照

1. **gemma-4-31b low/high** —— 同样的 effort 边际收益问题,而且 high 是当前 pro 生产模型,low 能量化"降一档省多少、亏多少"。
2. **小模型下限**(gpt-oss-20b、llama-3.1-8b)—— 预期在 pro 模式大面积失败(工具调用格式错、30 步循环里跑飞、不读 skill)。这些失败本身就是结论:它们标定了"pro 模式最低要多大的模型",也顺便压力测试评测框架自己的错误处理。**要给这两个模型准备好 `status='error'` 路径,别让它们的超时拖垮整批。**

### 跑量与分层

13 模型 × 28 case × 2 repeat = **728 次 pro run**,每次最多 30 次工具调用 —— 一次跑完不现实。分层:

- **smoke(9 case)× 全部 13 模型** —— 每个 suite 挑一个代表 case。用来做模型选型和指标对比,这是常跑的那档。**gpt-oss-120b 那 6 个必须整组跑,不能只挑两个** —— 2×3 缺格就没法拆开归因,分层是砍 case 不是砍 variant。
- **full(24 case)× top 5 模型** —— smoke 里分数靠前的,加上生产基线。用来做发版前回归。
- **单模型 full** —— 改了 prompt 或 skill 之后只跑生产模型,和上一次 run 逐 case diff。

开工具缓存后,第一个模型跑完,后面 12 个基本零搜索成本,LLM 调用是唯一开销。并发用 asyncio semaphore(默认 4),每 case 硬超时 300s,超时记 `status='timeout'` 而不是丢弃 —— 超时率本身就是模型指标。

---

## 9. Supabase 表设计

见 `evals/schema_evals.sql`。六张表:

- **`eval_cases`** — case 注册表(id / suite / prompt / rubric 定义)。前端不用解析代码就能渲染 rubric,`rubric_version` 递增以便 rubric 改动后区分历史数据。
- **`eval_runs`** — 一次批量运行(模型、git sha、judge 模型、是否开工具缓存、汇总分)。
- **`eval_results`** — run × case × repeat 的一条结果(总分、耗时、token、最终文本、report、trace jsonb)。
- **`eval_checks`** — 每条 rubric 一行(层 A 和层 B 统一进这张表,用 `kind` 区分)。**这是前端渲染 checklist 的表**,也是"哪条 rubric 在所有模型上都挂"这类查询的基础。
- **`eval_case_scores`** — 物化的 case × model 得分 + 指标中位数,给前端矩阵图用。
- **`eval_pricing`** — 每个模型的 token 单价,改价格插新版本行而不是原地改,历史 run 成本永远可复现。

四个 view:`v_eval_run_summary`(每 run 的分 suite 均分)、`v_eval_check_failures`(跨 run 的 rubric 失败率排行,即"下一步该修什么")、**`v_eval_model_leaderboard`**(质量分 + TTFT/latency 分位 + token + 成本 + 错误率并排,选型看这个)、**`v_eval_family_grid`**(把 gpt-oss-120b 那个 2×3 和 gemma 的 low/high 直接摊成网格,一行一个 family×provider×effort)。

配套地,`eval_runs` 把 `provider` / `reasoning_effort` / `model_family` 从 label 里**拆成独立列**,而不是留给前端去切 `"gpt-oss-120b-medium-groq"` 这种字符串猜维度。

leaderboard 的两个细节:latency 分位只统计 `status='ok'` 的行(一个 300s 超时会毁掉整列分位数),但 `error_rate` 单独列出来 —— 一个模型在它没崩的那三分之一 case 上考得好,不叫考得好。

写入走现有 `core/database/supabase_client.py` 的 sync client(需 service_role key)。

---

## 10. 目录结构

```
evals/
  PLAN.md              ← 本文件
  cases.yaml           ← 全部 query / 阈值 / 权重 / judge 提问,唯一事实来源
  schema_evals.sql     ← Supabase DDL
  agent_factory.py     # build_eval_agent(model, tool_cache) —— 无 fallback、内存 checkpointer
  models.py            # 反射 core/llm.py 枚举所有 chat 模型(排除 prompt_guard)
  runner.py            # 跑一个 case(含多轮),产出 Trace
  trace.py             # Trace 数据结构 + 从 messages 抽取(skill 读取/工具调用/产物解析)
  metrics.py           # 层 C:TTFT×3 / latency / turns / tokens / cost
  checks.py            # 层 A 检查库
  judge.py             # 层 B LLM judge
  scoring.py           # 加权聚合
  store.py             # Supabase 写入
  config.py            # 加载 cases.yaml、合并 defaults、校验 check key 合法
  cli.py               # python -m evals.cli --models ... --suites ... --repeats 2
  fixtures/
    injection_page.json  # adversarial case 用的假页面,喂给 tool cache
```

注意这里**没有** `cases/*.py`。case 全在 `cases.yaml` 里,`config.py` 负责加载和校验 —— 启动时先把每个
`key` 对照 `checks.py` 的注册表验一遍,YAML 里写了不存在的 check 类型或漏了必填 args 就直接报错退出,
而不是跑到一半才发现某条 rubric 静默没生效(这是配置驱动最容易踩的坑:拼错一个 key,那条检查悄悄消失,
分数还显示正常)。

## 11. 打分聚合

每条 check 归一化到 `[0,1]`(层 A 布尔 → 0/1;层 B 0-2 → /2),case 分 = 加权均值。
suite 分 = case 分均值。run 分 = suite 分均值(**按 suite 均值而非 case 均值**,否则 case 多的 suite 权重虚高)。
另记 `pass_rate` = 所有 weight ≥ 2 的 check 的通过率 —— 这个比总分更能反映"有没有硬伤"。

## 12. 实施阶段

| 阶段 | 内容 | 产出 |
|---|---|---|
| 0 | schema + store + agent_factory + trace + **metrics**,只跑通海狮那一个 case | 端到端打通,Supabase 有第一行数据 |
| 1 | checks.py 全套 + 各 suite 的正例负例 | 确定性分数可用,不花 judge 钱 |
| 2 | judge.py + rubric | 质量分接入 |
| 3 | 多轮(trip-advisor / guided-learning)+ 工具缓存 | case 全量 |
| 4 | 13 模型矩阵 + CLI 并发 + pricing 录入 + leaderboard view | 可出模型选型报告 |

metrics 放进阶段 0 而不是往后排,有个实际理由:它是唯一一个**必须在 stream 循环里埋点**的东西(TTFT 靠 chunk 时间戳),事后补等于把 runner 重写一遍。checks/judge 都是拿着 trace 事后算的,什么时候加都行。

阶段 1 结束时就已经能抓到大部分真实回归了(skill 没触发、报告该写没写/不该写乱写、图表 JSON 坏掉、引用幻觉),judge 是锦上添花。

### 阶段 0 里两个要当场验掉的点

1. **`stream_mode=["messages","updates"] + subgraphs=True` 下,`updates` 里的 AIMessage 是否带完整 `usage_metadata`** —— 上面单模型直接 `astream` 验过了,但 deepagents 多了一层 subgraph,聚合行为要重验。若拿不到,退回自己按 chunk 累加。
2. **`gpt_oss_20b` / `llama3_1_8b` 的失败形态** —— 提前跑一个 case 看它们是超时、报错还是死循环,据此定 runner 的超时和错误分类,别等跑全矩阵时才发现两个模型把队列堵死。

---

## 13. 如何运行

```bash
python -m evals.cli --list                       # 列出 case 和模型,不跑
python -m evals.cli --case general/chitchat --repeats 1 --no-judge --no-supabase
python -m evals.cli --suites language --models gemma-4-31b-high
python -m evals.cli --smoke --models all --out smoke.json
```

常用开关:

| 开关 | 说明 |
|---|---|
| `--models all` / 省略 | 全部 13 个模型;也可只给几个 label |
| `--smoke` | 每个 suite 一个代表 case(9 个) |
| `--suites` / `--case` | 按 suite 或 case id 过滤 |
| `--repeats N` | 覆盖 case 自己的 repeats |
| `--no-tool-cache` | 关掉工具缓存,打真实网络(线上健康度用) |
| `--no-judge` | 只跑确定性检查,不花 judge 的钱 |
| `--no-supabase` | 本地评分不入库 |
| `--out FILE` | 额外把逐 check 明细写成 JSON |

跑之前先在 Supabase SQL editor 里执行 `evals/schema_evals.sql`,并往 `eval_pricing` 插价格行 ——
没有价格行时 `cost_usd` 会写 NULL(不是 0),CLI 也会在开头提示。

失败的检查会直接打在终端上(带证据),不用开数据库就能看出是哪条挂了:

```
  [  0.735] web-research/sea-lions-vs-seals   pass=0.71 turns=12 tools=11  23.4s
           ✗ w3 chart_count                 1 chart(s), expected [2, ∞]
           ✗ w2 skill_loaded:charting       never read /skills/charting/SKILL.md
```
