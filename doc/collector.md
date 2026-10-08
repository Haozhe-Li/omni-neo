# Training-data collector

A password-protected page of the frontend (`/collect`) for producing SFT candidates by
hand: run the real agent under inputs a person chose (time, location, answer language,
memory), edit the final answer in a markdown editor, and file the conversation into
`sft_examples`. The thumbs-up capture is unchanged and writes the same table.

## Enabling it

| Where | Variable | Purpose |
| --- | --- | --- |
| backend | `COLLECTOR_API_KEY` | Unset = the feature is off (`/api/collector/*` answers 503). |
| frontend (server) | `COLLECTOR_API_KEY` | Same value; added by the Next proxy, never sent to browsers. |
| frontend | `COLLECTOR_PASSWORD` | Shared password for the login page. |
| frontend | `COLLECTOR_SESSION_SECRET` | HMAC key for the session cookie, >= 32 chars. |

Apply `schema.sql` (`python -m scripts.init_db`); it adds `sft_examples.source / edited /
original_final_text / collect_meta` and the `collector_turns` table, idempotently.

## Why the data matches production

The collector adds no path into the agent:

- `core/collector_schema.py` accepts only what the production client can send, in
  production's formats (`getLocalISOString`, `"City, Country (IP Approximate)"`, the
  settings-dialog language codes). `skill` takes only the three ids the chat's skill
  picker offers (`deep-research`, `trip-advisor`, `guided-learning`), resolved with the
  same `resolve_skill_name` `/chat` uses; attachments, source URLs and follow-up
  selections are not accepted.
- The system reminder and `<user_memory>` block come from `build_turn_context`
  (`core/utils/utils.py`), the function `POST /chat` calls.
- Generation is `_generate_background` from `core/routers/chat.py`, unmodified.
- The turn number is derived from the checkpoint (K-th user message = turn 2K-1), not
  taken from the client: it decides whether memory and the pre-flight scout run.
- What is stored is the real LangGraph checkpoint through `build_capture`.

`tests/test_collector_e2e.py` runs the same conversation through `/chat` and through the
collector and asserts the model receives byte-identical messages.

## Flow

`POST /threads` -> `POST /generate` (SSE, same events as `/chat`) -> `GET /threads/{id}/state`
(the answer as the checkpoint holds it) -> `PUT /threads/{id}/final` (edit; rewrites the
checkpoint's last answer in place, so a following turn reads the edit) -> `POST /generate`
again for another turn -> `POST /threads/{id}/submit`.

Submit files the conversation through the newest turn: **accepted** if any answer was
edited, **pending** if untouched. Only teacher models (`best`, `luna`) are accepted.
Memory can be set on the first turn only, as in production.

`build_dataset.py` keeps `<user_memory>` on `source = 'collector'` rows (the memory is
written by the annotator, not a real person's data).
