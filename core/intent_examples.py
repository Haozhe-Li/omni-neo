"""Anchor examples for core/intent_router.py — the only place to edit routes.

The router embeds every string here and routes a query to the label of its
nearest anchor. Changing anything below is enough: the embeddings are cached in
Redis under a hash of this content, so an edit simply misses the cache and
re-embeds on next start.

Labels are the scout's actions with the two weather actions merged (the router
decides *what kind* of turn this is; extracting a location or ticker is still the
LLM's job):

    about_omni | weather | stock | currency | web_search | direct_response

Writing good anchors:

- Narrow labels (weather, stock, currency, about_omni) form tight clusters, so a
  dozen varied phrasings in both languages is plenty. Vary the *shape* — a full
  question, a bare noun phrase, an implicit-location one — not just the city.
- The two broad labels (web_search, direct_response) have no centre. Cover the
  distinct sub-shapes instead of piling up near-duplicates, and expect them to
  match worse than the narrow ones.
- Hard negatives matter more than extra positives: the failures are topical
  look-alikes (a weather *question* that is not a weather lookup), and a few
  anchors of that shape fixed more than any change of algorithm did.
- Keep these independent of evals/scout_routing/cases.yaml. The eval asserts there
  is no exact overlap; near-paraphrases of eval queries would inflate it silently.
"""

from __future__ import annotations

INTENT_EXAMPLES: dict[str, list[str]] = {
    "about_omni": [
        "so who exactly am I talking to",
        "what is this assistant",
        "introduce yourself",
        "what's your name",
        "which AI model powers you",
        "who developed you and what company is behind you",
        "what features does Omni have",
        "how do Omni credits and pricing work",
        "can Omni run research on a schedule",
        "does this assistant keep a memory of me",
        "what file types can I upload here",
        "你是哪个公司做的",
        "你是谁开发的AI",
        "你都有哪些功能",
        "你用的是什么大模型",
        "Omni 怎么收费",
        "能不能设置定时任务让你每天调研",
        "你会记住我说过的话吗",
        "你支持上传哪些文件格式",
    ],
    "weather": [
        "weather in Paris right now",
        "is it going to rain this afternoon in Boston",
        "temperature in Singapore today",
        "what's the forecast for Toronto this week",
        "do I need an umbrella tomorrow in Seoul",
        "will it snow in Chicago this weekend",
        "how windy is it in Wellington",
        "is it sunny outside",
        "东京现在下雨吗",
        "广州明天的天气",
        "上海这周会降温吗",
        "今天外面热不热",
        "北京明天几度",
        "周末天气怎么样",
        "出门要不要带伞",
    ],
    "stock": [
        "Nvidia stock today",
        "how is Apple's share price doing",
        "Tesla shares after the earnings report",
        "what's the market cap of Microsoft",
        "is Amazon stock going up",
        "AMD quarterly earnings results",
        "how did the S&P 500 close",
        "Netflix stock performance this week",
        "英伟达今天股价",
        "特斯拉财报出来了吗，股价怎么样",
        "苹果公司最近股票表现",
        "微软的市值是多少",
        "腾讯控股股价",
        "美股今天收盘情况",
    ],
    "currency": [
        "how many pounds is 200 dollars",
        "USD to CNY exchange rate today",
        "convert 5000 yen to euros",
        "what's the dollar worth in Canadian dollars",
        "EUR to GBP",
        "current rate for Mexican pesos to USD",
        "100 Swiss francs in dollars",
        "美元兑人民币汇率",
        "300欧元是多少人民币",
        "一万日元换多少美元",
        "港币兑美元今天多少",
        "澳元对人民币的汇率",
        "换算一下500英镑等于多少欧元",
        "韩元兑换人民币",
    ],
    "web_search": [
        "who is the CEO of Anthropic",
        "what is a vector database",
        "how do transformers work in machine learning",
        "best laptops for programming",
        "what happened in the news today",
        "history of the Roman Empire",
        "how to learn Spanish quickly",
        "latest research on Alzheimer's treatment",
        "compare PostgreSQL and MySQL",
        "is intermittent fasting effective",
        "when is the next solar eclipse",
        "best restaurants in Chicago",
        "what are the symptoms of vitamin D deficiency",
        "how does inflation affect interest rates",
        "Kubernetes pod networking explained",
        "react server components tutorial",
        "谁发明了互联网",
        "量子计算是什么原理",
        "最好用的笔记软件有哪些",
        "怎么才能快速学会游泳",
        "日本旅游需要办什么签证",
        "新能源汽车电池技术最新进展",
        "华为最新手机评测",
        "二战的主要原因是什么",
        "如何申请美国绿卡",
        # Hard negatives: questions that mention weather, markets or money as a
        # *topic* are lookups, not widget requests. Without these the head sends
        # "how do tornadoes form" to the weather card.
        "how do tornadoes form",
        "what causes thunderstorms",
        "为什么会下雨",
        "what is a bond",
        "how do interest rates work",
        "货币政策是什么意思",
        "history of the US dollar",
        "为什么日元一直在贬值",
        "how does the stock market work",
        "股票投资入门怎么学",
        "how did Google get started",
        "谷歌是怎么发展起来的",
        "the economy of Germany explained",
        "best time to visit Thailand",
        "冬天适合去哪里旅游",
    ],
    "direct_response": [
        "translate this sentence into Spanish",
        "rewrite my paragraph to be more concise",
        "proofread this essay for grammar mistakes",
        "summarize the text I pasted below",
        "write a short poem about the ocean",
        "come up with five slogans for a bakery",
        "tell me a bedtime story about a dragon",
        "hey, how's it going",
        "good morning!",
        "write a SQL query that joins two tables",
        "debug this JavaScript function for me",
        "what's 15 percent of 240",
        "help me draft a resignation letter",
        "should I text her first or wait",
        "I feel stressed about work lately",
        "thanks, that was helpful",
        "把下面这段话翻译成日语",
        "帮我改一下这段文字的语气",
        "写一首关于春天的诗",
        "给我讲个笑话吧",
        "你好呀",
        "帮我写个python脚本批量重命名文件",
        "三百五十乘以十二等于几",
        "我最近压力很大不知道怎么办",
        "帮我起几个公司名字",
        "谢谢你",
        # Hard negatives: small talk that addresses "you" is not a question about Omni.
        "what are your plans for the evening",
        "do you like pizza",
        "你喜欢什么颜色",
        "do you have any hobbies",
        "你周末喜欢做什么",
    ],
}
