"""The options the intent router asks OpenAI's Decisions API to choose between.

`core/intent_router.py` sends the user's message with one multiple-choice
question; the answer is one of the `value`s below. This file is the only place
to edit routes, and the `description`s are the whole classifier — there are no
anchor examples any more, so what used to be taught by example (above all the
hard negatives, topical look-alikes that are NOT the narrow label) is written
into the descriptions instead.

The labels are the scout's actions with the two weather actions merged: the
router decides *what kind* of turn this is; extracting a location or a ticker is
still the scout LLM's job (core/context_enrichment.py).

    about_omni | weather | stock | currency | web_search | direct_response

plus one label per auto-triggerable skill, named ``skill:<skill-dir-name>``
(``SKILL_LABEL_PREFIX``). A skill label is decided by the router alone and loads
that skill exactly as if the user had picked it in the skill picker
(`requested_skill_enrichment` in core/context_enrichment.py). To make another skill
auto-triggerable, add a ``"skill:<name>"`` entry here — nothing else changes.

Order is not meaningful to the API, but the broad catch-all labels sit last.
Keep it independent of evals/scout_routing/cases.yaml: the eval measures these
descriptions on queries they were not written from.
"""

from __future__ import annotations

SKILL_LABEL_PREFIX = "skill:"

INSTRUCTIONS = (
    "Decide what kind of request this user message is, so the assistant can fetch "
    "the right context before answering. Pick the single best option. A topic that "
    "merely mentions weather, money or markets is not a live lookup: choose the "
    "option for what the user is actually asking for."
)

INTENT_CHOICES: dict[str, str] = {
    "weather": (
        "Asks for the current weather or an upcoming forecast for a place: temperature, "
        "rain, snow, wind, whether to bring an umbrella, including an implied place "
        "('is it sunny outside', '今天外面热不热'). NOT for how weather works "
        "('how do tornadoes form', '为什么会下雨'), climate or historical weather, or "
        "travel advice about a season ('best time to visit Thailand') — those are web_search."
    ),
    "stock": (
        "Asks about a specific publicly traded company's or market index's stock price, "
        "market cap, earnings results or market performance ('Nvidia stock today', "
        "'特斯拉财报出来了吗，股价怎么样', 'how did the S&P 500 close'). NOT for how markets "
        "or bonds work, investing basics, a company's history, or private companies — "
        "those are web_search."
    ),
    "currency": (
        "Asks to convert an amount between two national currencies or for an exchange "
        "rate ('how many pounds is 200 dollars', '美元兑人民币汇率', '一万日元换多少美元'). "
        "NOT for why a currency moves, monetary policy, a currency's history, or "
        "cryptocurrency — those are web_search."
    ),
    "about_omni": (
        "Asks about this assistant itself: who or what it is, its name, who built it or "
        "which company is behind it, which AI model powers it, what it can do, its "
        "features, pricing, credits, scheduled research, memory or file uploads "
        "('introduce yourself', '你是谁开发的', 'how do Omni credits work'). NOT small talk "
        "that merely says 'you' ('do you like pizza', '你周末喜欢做什么') and NOT questions "
        "about AI in general — those are direct_response and web_search."
    ),
    "skill:web-research": (
        "Explicitly asks for a deep, multi-step research job: a deep dive, a thorough "
        "well-sourced investigation, a comprehensive report, a literature review, "
        "'deeply research', '深度调研', '做一份详细的调研报告'. The signal is the request "
        "for depth, never the topic. A plain question about a topic, even one that "
        "uses the word research ('what does the research say about coffee and sleep', "
        "'quick search: who won the 2022 world cup', '帮我查一下今年高考时间'), is web_search."
    ),
    "web_search": (
        "A question about the world that a web lookup would answer: facts, people, "
        "companies, news, how things work, explanations, tutorials, recommendations, "
        "comparisons, local information, travel, health, history. Anything with a "
        "checkable answer belongs here however famous the subject ('who is the CEO of "
        "Anthropic', 'what is a vector database', '量子计算是什么原理')."
    ),
    "direct_response": (
        "Needs no lookup: translating, rewriting, proofreading or summarizing text the "
        "user supplied, creative writing, brainstorming names or slogans, coding help, "
        "arithmetic, drafting a letter, personal advice or feelings, small talk, "
        "greetings and thanks, or a message too vague to search for. NOT a question about "
        "the world ('who is X', 'what is X', 'is X any good') — those are web_search."
    ),
}
