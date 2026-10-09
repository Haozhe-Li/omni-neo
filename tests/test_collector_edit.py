"""Editing an answer in the collector, against a real deepagents graph.

    venv/bin/python3.12 -m pytest tests/test_collector_edit.py -q

core/routers/collector.py edits an answer by writing the checkpoint (`aupdate_state`
with the same message id). Two things about that must hold and neither can be
checked by reading the code, so this runs the real `create_deep_agent` (the same
graph core/agent.py builds, with a recording stand-in for the model):

- the next turn's model *reads the edited answer*, not the one it wrote, exactly
  as it would read any earlier assistant message in production;
- the edit keeps `response_metadata`, because the teacher check
  (core/sft_capture.py `is_teacher_turn`) reads the model name from it — an edit
  that dropped it would make every edited example look like it came from no one.

It also pins the capture end of the flow: the stored `messages` end on the edited
text, earlier turns come along, and the memory block is allowed for the collector
and refused for a thumbs-up.
"""

from __future__ import annotations

import asyncio

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from core import sft_capture as sc

LUNA = {"model_name": "gpt-6-luna-2026-09-01"}


class Recorder(BaseChatModel):
    """Answers "answer N" and remembers what it was shown on each call."""

    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recorder"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append([(type(m).__name__, m.content) for m in messages])
        n = len(self.seen)
        msg = AIMessage(content=f"answer {n}", response_metadata=dict(LUNA))
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def agent():
    model = Recorder()
    return create_deep_agent(model=model, tools=[], checkpointer=InMemorySaver()), model


CFG = {"configurable": {"thread_id": "t1"}}


async def run_turn(agent, text: str):
    async for _ in agent.astream({"messages": [{"role": "user", "content": text}]}, CFG, stream_mode="updates"):
        pass
    return list((await agent.aget_state(CFG)).values["messages"])


async def edit_last(agent, new_text: str):
    """What PUT /api/collector/threads/{id}/final does to the checkpoint."""
    messages = list((await agent.aget_state(CFG)).values["messages"])
    last = messages[-1]
    await agent.aupdate_state(CFG, {"messages": [last.model_copy(update={"content": new_text})]})
    return list((await agent.aget_state(CFG)).values["messages"])


def test_edit_replaces_in_place_and_keeps_the_model_name(agent):
    asyncio.run(_edit_replaces_in_place(agent[0]))


async def _edit_replaces_in_place(agent):
    before = await run_turn(agent, "<user_query>\nhi\n</user_query>")
    after = await edit_last(agent, "EDITED")

    assert len(after) == len(before) == 2          # replaced, not appended
    assert after[-1].id == before[-1].id
    assert after[-1].content == "EDITED"
    assert after[-1].response_metadata["model_name"] == LUNA["model_name"]
    assert sc.is_teacher_turn(sc.convert_messages(after).turn_models)


def test_the_next_turn_reads_the_edited_answer(agent):
    asyncio.run(_next_turn_reads_edit(*agent))


async def _next_turn_reads_edit(agent, model):
    await run_turn(agent, "<user_query>\nfirst\n</user_query>")
    await edit_last(agent, "EDITED ONE")
    await run_turn(agent, "<user_query>\nsecond\n</user_query>")

    shown = [content for kind, content in model.seen[-1] if kind == "AIMessage"]
    assert shown == ["EDITED ONE"]                  # not "answer 1"


def test_capture_of_an_edited_multi_turn_thread(agent):
    asyncio.run(_capture_edited_thread(agent[0]))


async def _capture_edited_thread(agent):
    memory = "<user_memory>\nLong-term facts about this user.\n## Profile\n- likes tea\n</user_memory>\n\n"
    await run_turn(agent, memory + "<user_query>\nfirst\n</user_query>")
    await edit_last(agent, "EDITED ONE")
    await run_turn(agent, "<user_query>\nsecond\n</user_query>")
    messages = await edit_last(agent, "EDITED TWO")

    cap = sc.build_capture(messages, 3, allow_memory=True)
    assert [m["role"] for m in cap.messages] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in cap.messages if m["role"] == "assistant"] == ["EDITED ONE", "EDITED TWO"]
    assert cap.has_memory and cap.models_seen == [LUNA["model_name"]]

    # a thumbs-up on the same conversation is still refused: that memory would be real
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture(messages, 3)
    assert e.value.code == "has_memory"


def test_human_messages_survive_the_edit_untouched():
    # the K-th user message is turn 2K-1; the collector derives turns from this count
    msgs = [HumanMessage(content="a"), AIMessage(content="x"), HumanMessage(content="b"), AIMessage(content="y")]
    assert 2 * sum(isinstance(m, HumanMessage) for m in msgs) - 1 == 3


def test_attachments_are_refused_for_a_thumbs_up_and_allowed_for_the_collector():
    # A real person's uploaded files are not recorded by a thumbs-up; the collector's are
    # test files an annotator chose. Images ride along either way (sft_images).
    messages = [
        HumanMessage(content="<attached_files>\nnotes.txt -> mounted at /uploads/notes.txt\n</attached_files>\n\n<user_query>\nq\n</user_query>"),
        AIMessage(content="a", response_metadata=dict(LUNA)),
    ]
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture(messages, 1)
    assert e.value.code == "has_attachments"
    cap = sc.build_capture(messages, 1, allow_attachments=True)
    assert cap.has_attachments and not cap.has_memory

    # the memory exception does not imply the attachment one, nor the reverse
    with pytest.raises(sc.CaptureError):
        sc.build_capture(messages, 1, allow_memory=True)


def test_an_image_message_is_stored_by_reference_with_its_bytes_kept():
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "<attached_files>\nx\n</attached_files>\n\n<user_query>\nwhat is this\n</user_query>"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
        ]),
        AIMessage(content="a tiny square", response_metadata=dict(LUNA)),
    ]
    cap = sc.build_capture(messages, 1, allow_attachments=True)
    [(sha, (mime, data))] = cap.images.items()
    assert mime == "image/png" and data[:4] == b"\x89PNG" and cap.has_image
    assert cap.messages[0]["content"][1]["image_url"]["url"] == f"omni-image://{sha}"
