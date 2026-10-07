"""The interactive agent's tool budget, offline: messages in, a decision out.

    venv/bin/python3.12 -m pytest tests/test_tool_budget.py -q

`forced_reason` is the whole policy (core/tool_budget.py); the middleware only
acts on it. What this cannot show is how real providers react to a request that
carries tool-call history but no tools — that is checked against live models
separately, and a fake handler here says nothing about it.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from core import tool_budget as tb


def human(text="q"):
    return HumanMessage(content=text)


def call(name="web_search", args=None, id_=None, i=[0]):
    i[0] += 1
    return {"id": id_ or f"c{i[0]}", "name": name, "args": args if args is not None else {"query": f"q{i[0]}"}, "type": "tool_call"}


def ai(*calls):
    return AIMessage(content="", tool_calls=list(calls))


def results(*calls):
    return [ToolMessage(content="r", tool_call_id=c["id"]) for c in calls]


def turn_with_calls(n, *, args=None, name="web_search"):
    """A user message followed by `n` single-call model/tool round trips."""
    msgs = [human()]
    for _ in range(n):
        c = call(name, args)
        msgs += [ai(c), *results(c)]
    return msgs


def load_skill(name):
    c = call("read_file", {"file_path": f"/skills/{name}/SKILL.md"})
    return [ai(c), *results(c)]


# ── rule 1: the budget ──────────────────────────────────────────────────────


def test_a_plain_turn_gets_ten_calls():
    assert tb.forced_reason(turn_with_calls(9)) is None
    assert tb.forced_reason(turn_with_calls(10)) == "limit"


def test_a_loaded_research_skill_raises_it_to_thirty():
    base = [human(), *load_skill("web-research")]
    assert tb.forced_reason(base + turn_with_calls(29)[1:]) is None
    assert tb.forced_reason(base + turn_with_calls(30)[1:]) == "limit"


@pytest.mark.parametrize("skill", ["web-research", "trip-advisor"])
def test_both_research_skills_count(skill):
    msgs = [human(), *load_skill(skill), *turn_with_calls(15)[1:]]
    assert tb.run_limit_for(msgs) == 30 and tb.forced_reason(msgs) is None


@pytest.mark.parametrize("skill", ["ask-question", "mapping", "report-writing", "charting", "guided-learning", "draft-email", "about-omni"])
def test_other_skills_do_not_raise_the_limit(skill):
    msgs = [human(), *load_skill(skill), *turn_with_calls(10)[1:]]
    assert tb.run_limit_for(msgs) == 10 and tb.forced_reason(msgs) == "limit"


def test_loading_a_skill_does_not_spend_budget():
    msgs = [human(), *load_skill("ask-question"), *turn_with_calls(9)[1:]]
    assert tb.forced_reason(msgs) is None  # 9 searches + 1 skill read: still under 10
    assert tb.forced_reason(msgs + turn_with_calls(1)[1:]) == "limit"


def test_a_skill_handed_over_by_the_app_counts_without_any_tool_call():
    picked = (
        "<requested_skill>\nweb-research\n</requested_skill>\n\n<context_enrichment>\n"
        "The web-research skill was already loaded for you, because the user explicitly picked it before asking. ..."
        "\n</context_enrichment>\n\n<user_query>q</user_query>"
    )
    msgs = [HumanMessage(content=picked), *turn_with_calls(20)[1:]]
    assert tb.run_limit_for(msgs) == 30 and tb.forced_reason(msgs) is None
    routed = "The trip-advisor skill was already loaded for you, because this request is the kind it exists for."
    assert tb.run_limit_for([HumanMessage(content=routed)]) == 30
    only_tag = "<requested_skill>\ntrip-advisor\n</requested_skill>\n<user_query>q</user_query>"
    assert tb.run_limit_for([HumanMessage(content=only_tag)]) == 30


def test_a_skill_loaded_in_an_earlier_turn_still_counts_but_its_calls_do_not():
    earlier = [human("first"), *load_skill("web-research"), *turn_with_calls(25)[1:], AIMessage(content="answer")]
    now = earlier + [human("follow up"), *turn_with_calls(3)[1:]]
    assert tb.run_limit_for(now) == 30          # still in the model's context
    assert tb.forced_reason(now) is None        # and the 25 earlier calls are not this turn's


def test_only_read_file_opens_a_skill():
    msgs = [human(), ai(call("web_search", {"query": "/skills/web-research/ notes"}))]
    assert tb.skills_loaded(msgs) == set()
    assert tb.skills_loaded([human(), ai(call("ls", {"path": "/skills/"}))]) == set()


def test_parallel_calls_each_count():
    c = [call() for _ in range(10)]
    assert tb.forced_reason([human(), ai(*c), *results(*c)]) == "limit"


# ── rule 2: loops ───────────────────────────────────────────────────────────


def test_three_identical_calls_in_a_row_force_an_answer():
    same = {"query": "weather paris"}
    assert tb.forced_reason(turn_with_calls(2, args=same)) is None
    assert tb.forced_reason(turn_with_calls(3, args=same)) == "repeat"


def test_argument_order_does_not_disguise_a_repeat():
    msgs = [human()]
    for args in ({"a": 1, "b": 2}, {"b": 2, "a": 1}, {"a": 1, "b": 2}):
        c = call("fetch_url", args)
        msgs += [ai(c), *results(c)]
    assert tb.forced_reason(msgs) == "repeat"


def test_different_arguments_or_tools_are_not_a_repeat():
    assert tb.forced_reason(turn_with_calls(1, args={"q": "a"}) + turn_with_calls(1, args={"q": "b"})[1:] + turn_with_calls(1, args={"q": "a"})[1:]) is None
    mixed = [human()]
    for name in ("web_search", "fetch_url", "web_search"):
        c = call(name, {"query": "x"})
        mixed += [ai(c), *results(c)]
    assert tb.forced_reason(mixed) is None


def test_the_repeat_must_be_the_most_recent_three():
    msgs = turn_with_calls(3, args={"q": "same"})
    c = call("web_search", {"q": "different"})
    assert tb.forced_reason(msgs + [ai(c), *results(c)]) is None


def test_three_identical_calls_inside_one_message_count():
    c = [call("web_search", {"q": "x"}) for _ in range(3)]
    assert tb.forced_reason([human(), ai(*c), *results(*c)]) == "repeat"


def test_a_repeat_in_an_earlier_turn_does_not_count_now():
    msgs = turn_with_calls(3, args={"q": "x"}) + [AIMessage(content="done"), human("next")]
    assert tb.forced_reason(msgs) is None


def test_repeating_a_skill_load_is_still_a_loop():
    msgs = [human(), *load_skill("web-research"), *load_skill("web-research"), *load_skill("web-research")]
    assert tb.forced_reason(msgs) == "repeat"


# ── the middleware ──────────────────────────────────────────────────────────


class FakeTool:
    name = "web_search"


def request(messages):
    return ModelRequest(
        model=None, messages=messages, system_message=SystemMessage(content="SYS"),
        tools=[FakeTool()], tool_choice={"type": "auto"}, state={"messages": messages},
    )


def run(messages, reply):
    seen = {}

    def handler(req):
        seen["req"] = req
        return reply

    return tb.ToolBudgetMiddleware().wrap_model_call(request(messages), handler), seen["req"]


def test_normal_calls_pass_through_untouched():
    reply = ModelResponse(result=[AIMessage(content="ok")])
    out, req = run(turn_with_calls(2), reply)
    assert out is reply and req.tools and req.tool_choice == {"type": "auto"}
    assert "no longer call tools" not in str(req.system_message.content)
    assert "tool_budget" not in reply.result[0].response_metadata


def test_a_forced_call_goes_out_with_no_tools_and_the_note():
    out, req = run(turn_with_calls(10), ModelResponse(result=[AIMessage(content="Here is my answer.")]))
    assert req.tools == [] and req.tool_choice is None
    assert "no longer call tools" in str(req.system_message.content) and "SYS" in str(req.system_message.content)
    assert out.result[0].response_metadata["tool_budget"] == "forced:limit"
    assert out.result[0].content == "Here is my answer."


def test_tool_calls_the_model_emits_anyway_are_dropped():
    # The tools node still knows every tool; a call that slipped through would run.
    sneaky = AIMessage(content="Let me check one more thing.", tool_calls=[call()])
    out, _ = run(turn_with_calls(3, args={"q": "x"}), ModelResponse(result=[sneaky]))
    msg = out.result[0]
    assert msg.tool_calls == [] and msg.content == "Let me check one more thing."
    assert msg.response_metadata["tool_budget"] == "forced:repeat"


def test_an_empty_forced_answer_gets_a_fallback_rather_than_silence():
    out, _ = run(turn_with_calls(10), ModelResponse(result=[AIMessage(content="", tool_calls=[call()])]))
    assert out.result[0].content == tb.FALLBACK_ANSWER and out.result[0].tool_calls == []


def test_a_model_that_answers_only_with_tool_calls_gets_one_retry_that_works():
    replies = [
        ModelResponse(result=[AIMessage(content="", tool_calls=[call()])]),
        ModelResponse(result=[AIMessage(content="Plain text this time.")]),
    ]
    seen = []

    def handler(req):
        seen.append(req)
        return replies[len(seen) - 1]

    out = tb.ToolBudgetMiddleware().wrap_model_call(request(turn_with_calls(10)), handler)
    assert len(seen) == 2
    assert seen[1].messages[-1].content == tb.RETRY_NUDGE and seen[1].tools == []
    assert out.result[0].content == "Plain text this time." and out.result[0].tool_calls == []
    assert out.result[0].response_metadata["tool_budget"] == "forced:limit"


def test_the_retry_happens_once_not_forever():
    calls = []

    def handler(req):
        calls.append(1)
        return ModelResponse(result=[AIMessage(content="", tool_calls=[call()])])

    out = tb.ToolBudgetMiddleware().wrap_model_call(request(turn_with_calls(10)), handler)
    assert len(calls) == 2 and out.result[0].content == tb.FALLBACK_ANSWER


def test_a_forced_answer_with_text_is_not_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return ModelResponse(result=[AIMessage(content="Fine.", tool_calls=[call()])])

    out = tb.ToolBudgetMiddleware().wrap_model_call(request(turn_with_calls(10)), handler)
    assert len(calls) == 1 and out.result[0].content == "Fine." and out.result[0].tool_calls == []


def test_a_bare_aimessage_result_is_handled_too():
    out, _ = run(turn_with_calls(10), AIMessage(content="plain"))
    assert out.response_metadata["tool_budget"] == "forced:limit"


def test_async_path_behaves_the_same():
    seen = {}

    async def handler(req):
        seen["req"] = req
        return ModelResponse(result=[AIMessage(content="done")])

    out = asyncio.run(tb.ToolBudgetMiddleware().awrap_model_call(request(turn_with_calls(10)), handler))
    assert seen["req"].tools == [] and out.result[0].response_metadata["tool_budget"] == "forced:limit"
