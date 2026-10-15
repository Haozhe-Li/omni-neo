"""What the step trace shows a finished retrieval tool as having read, offline.

    venv/bin/python3.12 -m pytest tests/test_tool_results.py -q
"""

from __future__ import annotations

import json

from core.tool_results import MAX_RESULTS, tool_result_pages


def test_search_list_keeps_order_and_drops_content():
    results = [
        {"title": "A", "url": "https://a.com/x", "content": "long", "n": 1},
        {"title": "B", "url": "https://b.org/y", "content": "long", "n": 2},
    ]
    assert tool_result_pages(json.dumps(results)) == [
        {"title": "A", "url": "https://a.com/x"},
        {"title": "B", "url": "https://b.org/y"},
    ]


def test_fetch_url_single_dict():
    page = {"title": "T", "url": "https://t.com", "content": "c", "n": 3}
    assert tool_result_pages(json.dumps(page)) == [{"title": "T", "url": "https://t.com"}]


def test_content_blocks_and_native_objects():
    blocks = [{"type": "text", "text": json.dumps([{"title": "A", "url": "https://a.com"}])}]
    assert tool_result_pages(blocks) == [{"title": "A", "url": "https://a.com"}]
    assert tool_result_pages([{"title": "A", "url": "https://a.com"}]) == [
        {"title": "A", "url": "https://a.com"}
    ]


def test_quote_resolves_through_its_citation():
    quote = json.dumps({"success": True, "data": {"symbol": "TSLA"}, "n": 4})
    lookup = {4: {"title": "Tesla (TSLA) stock quote", "url": "https://finance.yahoo.com/quote/TSLA"}}
    assert tool_result_pages(quote, lookup.get) == [
        {"title": "Tesla (TSLA) stock quote", "url": "https://finance.yahoo.com/quote/TSLA"}
    ]


def test_nothing_retrieved():
    # No-result / unavailable placeholders carry an empty url; errors and plain
    # strings are not JSON of the right shape. None of them are "pages read".
    assert tool_result_pages(json.dumps([{"title": "No results", "url": "", "content": ""}])) == []
    assert tool_result_pages("Error: boom") == []
    assert tool_result_pages(json.dumps({"error": "weather failed"})) == []
    assert tool_result_pages(None) == []


def test_dedupes_and_caps():
    dup = [{"title": "A", "url": "https://a.com"}] * 3
    assert len(tool_result_pages(json.dumps(dup))) == 1
    many = [{"title": str(i), "url": f"https://x.com/{i}"} for i in range(MAX_RESULTS + 5)]
    assert len(tool_result_pages(json.dumps(many))) == MAX_RESULTS
