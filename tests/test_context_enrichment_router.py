"""How enrich_context acts on the intent router: offline, the router, the scout LLM
and the tools are all fakes.

    venv/bin/python3.12 -m pytest tests/test_context_enrichment_router.py -q

The contract under test: the router decides alone for web_search / about_omni /
direct_response; weather, stock and currency (arguments needed), long web_search
queries (needs a short reformulation), "unsure", and any router failure all go to the
scout LLM. Nothing the router does may raise into the turn.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

warnings.filterwarnings("ignore")

from core import context_enrichment as ce  # noqa: E402
from core.intent_router import RouteResult, RouterError  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def result(label, score=0.9):
    return RouteResult(label=label, score=score)


class FakeRouter:
    def __init__(self, res=None, exc=None, delay=0.0):
        self.res, self.exc, self.delay, self.calls = res, exc, delay, 0

    async def route(self, query):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.res


class Harness:
    """Patches the router in, counts scout LLM calls and tool runs."""

    def __init__(self, monkeypatch, router, scout_decision=None):
        self.llm_calls, self.tool_calls = [], []
        self.routed_labels = []

        async def no_router(query):
            raise RouterError("router unavailable")

        monkeypatch.setattr(ce, "router_route", router.route if router is not None else no_router)

        async def fake_classify(query, loc=None, dt=None, timeout=0, routed_label=None):
            self.llm_calls.append(query)
            self.routed_labels.append(routed_label)
            return scout_decision

        monkeypatch.setattr(ce, "classify", fake_classify)

        def fake_action(name):
            real = ce._ACTIONS[name]

            async def fake_run(d):
                self.tool_calls.append((name, d))
                return "BODY", None

            return ce._Action(tool=real.tool, args=real.args, run=fake_run, ready=real.ready)

        for name in ("web_search", "stock", "weather_current", "weather_forecast", "currency"):
            monkeypatch.setitem(ce._ACTIONS, name, fake_action(name))
        monkeypatch.setattr(ce, "_about_omni_text", lambda: "OMNI SKILL BODY")


def test_web_search_decided_by_router_skips_the_llm_and_searches_the_query_verbatim(monkeypatch):
    h = Harness(monkeypatch, FakeRouter(result("web_search")))
    out = run(ce.enrich_context("  what is a vector database  "))
    assert h.llm_calls == []
    assert h.tool_calls[0][0] == "web_search" and h.tool_calls[0][1].search_query == "what is a vector database"
    assert out.action == "web_search" and "BODY" in out.text
    assert out.events[0] == {
        "type": "tool_call", "tool": "web_search", "args": {"query": "what is a vector database"},
        "id": ce._SCOUT_CALL_ID,
    }


def test_direct_response_enriches_nothing_and_does_not_call_the_llm(monkeypatch):
    h = Harness(monkeypatch, FakeRouter(result("direct_response")))
    out = run(ce.enrich_context("write a haiku about autumn"))
    assert h.llm_calls == [] and h.tool_calls == []
    assert out.text == "" and out.events == [] and out.sources == []


def test_about_omni_injects_the_skill_without_the_llm(monkeypatch):
    h = Harness(monkeypatch, FakeRouter(result("about_omni")))
    out = run(ce.enrich_context("who are you"))
    assert h.llm_calls == [] and out.action == "about_omni" and "OMNI SKILL BODY" in out.text


@pytest.mark.parametrize("label", ["weather", "stock", "currency"])
def test_widget_labels_still_go_to_the_llm_because_it_extracts_the_arguments(monkeypatch, label):
    decision = ce.EnrichmentDecision(action="stock", ticker="NVDA")
    h = Harness(monkeypatch, FakeRouter(result(label)), scout_decision=decision)
    out = run(ce.enrich_context("nvidia stock today"))
    assert h.llm_calls == ["nvidia stock today"]
    assert h.routed_labels == [label]  # the scout is told what the router decided
    assert h.tool_calls == [("stock", decision)] and out.action == "stock"


def test_long_web_search_query_goes_to_the_llm_for_a_short_reformulation(monkeypatch):
    long_q = " ".join(["word"] * (ce._SHORTCUT_WORD_LIMIT + 2))
    decision = ce.EnrichmentDecision(action="web_search", search_query="short query")
    h = Harness(monkeypatch, FakeRouter(result("web_search")), scout_decision=decision)
    run(ce.enrich_context(long_q))
    assert h.llm_calls == [long_q] and h.routed_labels == ["web_search"]
    assert h.tool_calls[0][1].search_query == "short query"  # the LLM's, not the pasted paragraph


def test_router_unavailable_falls_back_to_the_llm_with_no_prior(monkeypatch):
    decision = ce.EnrichmentDecision(action="direct_response")
    h = Harness(monkeypatch, None, scout_decision=decision)
    out = run(ce.enrich_context("hello"))
    assert h.llm_calls == ["hello"] and h.routed_labels == [None] and out.text == ""


def test_router_error_falls_back_to_the_llm_and_never_raises(monkeypatch):
    decision = ce.EnrichmentDecision(action="direct_response")
    h = Harness(monkeypatch, FakeRouter(exc=RouterError("decisions API down")), scout_decision=decision)
    out = run(ce.enrich_context("hello"))
    assert h.llm_calls == ["hello"] and out.text == ""


def test_slow_router_is_cut_off_and_falls_back_to_the_llm(monkeypatch):
    monkeypatch.setattr(ce, "_ROUTER_TIMEOUT_S", 0.05)
    decision = ce.EnrichmentDecision(action="direct_response")
    h = Harness(monkeypatch, FakeRouter(result("web_search"), delay=1.0), scout_decision=decision)
    out = run(ce.enrich_context("what is rust"))
    assert h.llm_calls == ["what is rust"] and h.tool_calls == [] and out.text == ""


def test_llm_failure_after_a_router_error_is_still_just_no_enrichment(monkeypatch):
    h = Harness(monkeypatch, FakeRouter(exc=RouterError("down")), scout_decision=None)
    out = run(ce.enrich_context("anything"))
    assert h.llm_calls == ["anything"] and out.text == "" and out.events == []


def test_the_router_is_not_asked_about_queries_the_length_gate_already_drops(monkeypatch):
    router = FakeRouter(result("web_search"))
    h = Harness(monkeypatch, router)
    out = run(ce.enrich_context("word " * (ce._WORD_LIMIT + 5)))
    assert router.calls == 0 and h.llm_calls == [] and out.text == ""


def test_empty_query_is_a_no_op(monkeypatch):
    router = FakeRouter(result("web_search"))
    Harness(monkeypatch, router)
    assert run(ce.enrich_context("   ")).text == "" and router.calls == 0


# ── skill labels ────────────────────────────────────────────────────────────

SKILL_BODY = "WEB RESEARCH WORKFLOW"


def with_skill_file(monkeypatch, path="/skills/web-research/SKILL.md"):
    monkeypatch.setattr(
        ce, "SKILL_FILES", {path: {"content": f"---\nname: web-research\n---\n\n{SKILL_BODY}"}}
    )


def test_skill_label_loads_the_skill_without_the_llm_or_a_search(monkeypatch):
    with_skill_file(monkeypatch)
    h = Harness(monkeypatch, FakeRouter(result("skill:web-research")))
    out = run(ce.enrich_context("深度研究一下固态电池"))
    assert h.llm_calls == [] and h.tool_calls == []
    assert out.action == "routed_skill" and out.skill == "web-research"
    assert SKILL_BODY in out.text and "kind it exists for" in out.text
    assert out.sources == []


def test_routed_skill_emits_the_same_read_file_event_as_a_picked_one(monkeypatch):
    with_skill_file(monkeypatch)
    Harness(monkeypatch, FakeRouter(result("skill:web-research")))
    routed = run(ce.enrich_context("deep research on vector databases"))
    picked = ce.requested_skill_enrichment("web-research")
    assert routed.events == picked.events == [
        {"type": "tool_call", "tool": "read_file", "args": {"file_path": "/skills/web-research/SKILL.md"}}
    ]
    assert picked.skill == "web-research" and "explicitly picked" in picked.text
    # Same body either way; only the sentence saying why differs.
    assert routed.text.split("\n\n", 1)[1] == picked.text.split("\n\n", 1)[1]


def test_skill_label_with_a_missing_skill_file_falls_back_to_the_scout(monkeypatch):
    monkeypatch.setattr(ce, "SKILL_FILES", {})
    decision = ce.EnrichmentDecision(action="web_search", search_query="solid state batteries")
    h = Harness(monkeypatch, FakeRouter(result("skill:web-research")), scout_decision=decision)
    out = run(ce.enrich_context("deep research solid state batteries"))
    assert h.llm_calls == ["deep research solid state batteries"]
    assert out.skill is None and out.action == "web_search"


def test_routed_skill_only_reads_skill_labels():
    assert ce.routed_skill(result("skill:web-research")) == "web-research"
    for r in (result("web_search"), None):
        assert ce.routed_skill(r) is None


def test_the_shipped_choices_define_a_web_research_skill_that_exists_on_disk():
    from pathlib import Path

    from core.intent_choices import INTENT_CHOICES, SKILL_LABEL_PREFIX

    skills = [k[len(SKILL_LABEL_PREFIX):] for k in INTENT_CHOICES if k.startswith(SKILL_LABEL_PREFIX)]
    assert "web-research" in skills
    for name in skills:
        assert (Path(__file__).parent.parent / "skills" / name / "SKILL.md").is_file(), name


def test_scout_search_reports_the_pages_it_retrieved_as_a_tool_result(monkeypatch):
    from core.utils.citations import register_citation, reset_citation_registry

    Harness(monkeypatch, FakeRouter(result("web_search")))
    real = ce._ACTIONS["web_search"]

    async def search_that_cites(d):
        register_citation("A", "https://a.com/x", "c")
        register_citation("Junk", "https://junk.biz", "c", credibility={"label": "junk", "reason": "r"})
        register_citation("B", "https://b.org/y", "c")
        return "BODY", None

    monkeypatch.setitem(
        ce._ACTIONS, "web_search",
        ce._Action(tool=real.tool, args=real.args, run=search_that_cites, ready=real.ready),
    )
    reset_citation_registry()
    out = run(ce.enrich_context("what is a vector database"))
    call, result_ev = out.events[0], out.events[1]
    assert result_ev == {
        "type": "tool_result", "id": call["id"], "tool": "web_search",
        "results": [{"title": "A", "url": "https://a.com/x"}, {"title": "B", "url": "https://b.org/y"}],
    }
