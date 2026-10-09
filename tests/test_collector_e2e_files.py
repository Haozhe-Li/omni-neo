"""Files, images and pinned URLs through the collector, next to POST /chat, byte for byte.

    TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:5432/omni_test \\
    TEST_REDIS_URL=redis://127.0.0.1:6379/15 \\
        venv/bin/python3.12 -m pytest tests/test_collector_e2e_files.py -q

Same opt-in stack as test_collector_e2e.py, plus a real S3 API (moto, in-process) so
the upload is the production one end to end: a pending `user_files` row, a presigned
PUT the test performs over HTTP, a confirm that parses the stored object, and the
image bytes read back out of the bucket when the message is built.

The pinned-URL fetch is the one thing stubbed (it calls an external scraper); both
flows are asserted to hand it the same list.
"""

from __future__ import annotations

import base64
import os

import pytest

URL = os.getenv("TEST_DATABASE_URL")
REDIS = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not (URL and REDIS), reason="TEST_DATABASE_URL / TEST_REDIS_URL not set")

from tests.collector_stack import stack  # noqa: E402

GUEST = "guest_e2e-files"
LOC = "Shanghai, China (IP Approximate)"
DT = "2026-10-07T14:05:09+08:00"
# a real 1x1 PNG
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
NOTES = "Q3 revenue was 4.2M.\nChurn fell to 1.9%.\n".encode()
URLS = ["https://en.wikipedia.org/wiki/Rapid_transit", "https://example.com/a?b=1"]


@pytest.fixture
def s3(monkeypatch):
    """A real S3 API on localhost, wired in as the file parser's client."""
    import boto3
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=5557, verbose=False)
    server.start()
    try:
        client = boto3.client(
            "s3", endpoint_url="http://127.0.0.1:5557", aws_access_key_id="t", aws_secret_access_key="t",
            region_name="us-east-1",
        )
        client.create_bucket(Bucket="omni")
        import core.RAG.file_parser as fp

        monkeypatch.setattr(fp, "s3_client", client)
        monkeypatch.setenv("S3_BUCKET_NAME", "omni")
        yield client
    finally:
        server.stop()


def put(url: str, data: bytes, content_type: str) -> None:
    import httpx

    r = httpx.put(url, content=data, headers={"Content-Type": content_type})
    assert r.status_code == 200, r.text


def keys(s3) -> list[str]:
    return [o["Key"] for o in s3.list_objects_v2(Bucket="omni").get("Contents", [])]


def test_files_images_and_urls_match_chat(s3):
    with stack() as st:
        client, models, pg = st.client, st.models, st.pg
        human_texts = st.human_texts

        fetched: list[list[str]] = []

        def fake_fetch(urls):
            fetched.append(list(urls))
            return "PINNED PAGE TEXT for " + ", ".join(urls), {}, []

        st.stream_mod._fetch_source_urls = fake_fetch
        for t in ("user_memories", "collector_turns", "user_files"):
            pg.execute(f"DELETE FROM {t}")
        pg.execute("DELETE FROM sft_examples WHERE source = 'collector'")
        pg.execute("DELETE FROM user_usage WHERE user_id = %s", (GUEST,))

        guest = {"X-Guest-Id": GUEST}
        h = {"X-Collector-Key": "k", "X-Collector-Id": "alice"}

        # ── production: upload through /api/upload, then /chat ──────────────
        def prod_upload(thread_id, name, mime, data) -> str:
            r = client.post("/api/upload/url", headers=guest, json={
                "filename": name, "file_type": mime, "file_size_bytes": len(data), "thread_id": thread_id})
            assert r.status_code == 200, r.text
            put(r.json()["upload_url"], data, mime)
            assert client.post(f"/api/upload/confirm?file_id={r.json()['file_id']}", headers=guest).status_code == 200
            return r.json()["file_id"]

        def prod_chat(thread_id, query, files, urls, turn=1):
            payload = {
                "query": query, "thread_id": thread_id, "model": "best", "turn": turn,
                "personalization": {"response_language": "en", "user_local_datetime": DT, "user_location": LOC},
            }
            if files:
                payload["attached_file_ids"] = files
            if urls:
                payload["source_url"] = urls
            with client.stream("POST", "/chat", json=payload, headers=guest) as r:
                assert r.status_code == 200, r.read()
                body = "".join(r.iter_text())
            assert '"done"' in body, body[-300:]

        pt = client.get("/get_thread_id", headers=guest).json()
        pf1 = prod_upload(pt, "notes.txt", "text/plain", NOTES)
        pf2 = prod_upload(pt, "shot.png", "image/png", PNG)
        prod_chat(pt, "Summarise these and the page.", [{pf1: "notes.txt"}, {pf2: "shot.png"}], URLS)
        prod_msgs = human_texts(models["best"])
        prod_urls_fetched = list(fetched[-1])

        # ── collector: the same, through its own endpoints ──────────────────
        def col_thread() -> str:
            return client.post("/api/collector/threads", headers=h).json()["thread_id"]

        def col_upload(thread_id, name, mime, data, headers=h) -> str:
            r = client.post(f"/api/collector/threads/{thread_id}/uploads", headers=headers,
                            json={"filename": name, "file_type": mime, "file_size_bytes": len(data)})
            assert r.status_code == 200, r.text
            fid = r.json()["file_id"]
            assert fid.startswith("user_uploads/collector_")        # same key shape as a user's upload
            put(r.json()["upload_url"], data, mime)
            c = client.post(f"/api/collector/uploads/confirm?file_id={fid}", headers=headers)
            assert c.status_code == 200 and c.json()["status"] == "ready", c.text
            return fid

        def col_generate(thread_id, query, files, urls, **extra):
            body = {"query": query, "thread_id": thread_id, "model": "best",
                    "personalization": {"user_local_datetime": DT, "user_location": LOC, "response_language": "en"}, **extra}
            if files:
                body["attached_file_ids"] = files
            if urls:
                body["source_url"] = urls
            with client.stream("POST", "/api/collector/generate", json=body, headers=h) as r:
                assert r.status_code == 200, r.read()
                text = "".join(r.iter_text())
            assert '"done"' in text, text[-300:]

        ct = col_thread()
        cf1 = col_upload(ct, "notes.txt", "text/plain", NOTES)
        cf2 = col_upload(ct, "shot.png", "image/png", PNG)
        col_generate(ct, "Summarise these and the page.", [{cf1: "notes.txt"}, {cf2: "shot.png"}], URLS)
        col_msgs = human_texts(models["best"])

        # ── the claim ───────────────────────────────────────────────────────
        assert fetched[-1] == prod_urls_fetched == URLS, "the pinned URLs reach the fetch identically"
        assert len(col_msgs) == len(prod_msgs) == 1
        prod_content, col_content = prod_msgs[0], col_msgs[0]
        # an image makes the message a list of blocks: one text block and the inlined image
        assert isinstance(col_content, list) and [b["type"] for b in col_content] == ["text", "image_url"]
        # the file ids differ between the two runs (random uuids) but nothing else may:
        # the mount names, citation numbers, notes and the image bytes are all derived the same way
        def normalise(blocks, fids):
            out = []
            for b in blocks:
                b = dict(b)
                if b["type"] == "text":
                    for f in fids:
                        b["text"] = b["text"].replace(f, "<FILE>")
                out.append(b)
            return out

        assert normalise(col_content, [cf1, cf2]) == normalise(prod_content, [pf1, pf2])
        text = col_content[0]["text"]
        assert "<attached_files>" in text and "notes.txt -> mounted at /uploads/notes.txt, cite as [1]" in text
        assert "<context_enrichment>\nPINNED PAGE TEXT for " + ", ".join(URLS) in text
        assert col_content[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()

        # the turn's record keeps what was attached
        state = client.get(f"/api/collector/threads/{ct}/state", headers=h).json()
        t0 = state["turns"][0]
        assert [(a["filename"], a["category"]) for a in t0["attachments"]] == [("notes.txt", "document"), ("shot.png", "image")]
        assert t0["source_urls"] == URLS

        # ── submit: files the row with the image stored, attachments allowed ─
        r = client.post(f"/api/collector/threads/{ct}/submit", json={}, headers=h)
        assert r.status_code == 200, r.text
        row = pg.fetch_one("SELECT * FROM sft_examples WHERE thread_id = %s", (ct,))
        assert (row["source"], row["has_image"], row["has_attachments"], row["status"]) == ("collector", True, True, "pending")
        user_blocks = row["messages"][0]["content"]
        ref = user_blocks[1]["image_url"]["url"]
        assert ref.startswith("omni-image://")
        img = pg.fetch_one("SELECT mime, data FROM sft_images WHERE sha256 = %s", (ref.removeprefix("omni-image://"),))
        assert img["mime"] == "image/png" and bytes(img["data"]) == PNG
        assert row["collect_meta"]["turns"][0]["source_urls"] == URLS
        assert len(row["collect_meta"]["turns"][0]["attachments"]) == 2

        # ── a file-only turn, then someone else's file ─────────────────────
        ct2 = col_thread()
        f = col_upload(ct2, "notes.txt", "text/plain", NOTES)
        col_generate(ct2, "", [{f: "notes.txt"}], None)
        assert human_texts(models["best"])[0].rstrip().endswith("</attached_files>")   # no <user_query> block at all
        bob = {"X-Collector-Key": "k", "X-Collector-Id": "bob"}
        bt = client.post("/api/collector/threads", headers=bob).json()["thread_id"]
        bf = col_upload(bt, "notes.txt", "text/plain", NOTES, headers=bob)
        body = {"query": "hi", "thread_id": ct2, "personalization": {"user_local_datetime": DT},
                "attached_file_ids": [{bf: "notes.txt"}]}
        assert client.post("/api/collector/generate", json=body, headers=h).status_code == 422      # bob's file, alice's thread
        assert client.post(f"/api/collector/uploads/confirm?file_id={bf}", headers=h).status_code == 404

        # ── restart keeps the staged files; discard removes them ────────────
        rt = col_thread()
        rf = col_upload(rt, "notes.txt", "text/plain", NOTES)
        new_id = client.post(f"/api/collector/threads/{rt}/restart", headers=h).json()["thread_id"]
        assert new_id != rt and client.get(f"/api/collector/threads/{rt}/state", headers=h).status_code == 404
        rec = pg.fetch_one("SELECT thread_id FROM user_files WHERE file_id = %s", (rf,))
        assert rec["thread_id"] == new_id
        col_generate(new_id, "Use the notes.", [{rf: "notes.txt"}], None)           # the same id works in the new thread
        assert rf in keys(s3)
        assert client.delete(f"/api/collector/threads/{new_id}", headers=h).status_code == 200
        assert pg.fetch_one("SELECT 1 FROM user_files WHERE file_id = %s", (rf,)) is None
        assert rf not in keys(s3)                                                     # the object went with it
        assert cf1 in keys(s3) and cf2 in keys(s3)                                    # other conversations' files stay

        pg.execute("DELETE FROM sft_examples WHERE source = 'collector'")
        pg.execute("DELETE FROM user_files")
        pg.execute("DELETE FROM collector_turns")
