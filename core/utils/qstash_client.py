"""QStash integration: creates the recurring schedule behind a task, and
verifies the signature on incoming webhook deliveries.

The destination URL is this backend's own public `/scheduled_task/run`
endpoint (see core/routers/scheduled_tasks.py) — QStash calls it on the given
cron schedule with `{"task_id": ...}` as the JSON body every time.
"""

from __future__ import annotations

import logging
import os

from qstash import QStash, Receiver

logger = logging.getLogger(__name__)

_client: QStash | None = None
_receiver: Receiver | None = None


def _get_client() -> QStash:
    global _client
    if _client is None:
        _client = QStash(os.environ["QSTASH_TOKEN"])
    return _client


def _get_receiver() -> Receiver:
    global _receiver
    if _receiver is None:
        _receiver = Receiver(
            current_signing_key=os.environ["QSTASH_CURRENT_SIGNING_KEY"],
            next_signing_key=os.environ["QSTASH_NEXT_SIGNING_KEY"],
        )
    return _receiver


def verify_webhook_signature(*, signature: str, body: str, url: str) -> bool:
    try:
        _get_receiver().verify(signature=signature, body=body, url=url)
        return True
    except Exception as exc:
        logger.warning(f"[qstash_client] signature verification failed: {exc}")
        return False


def create_schedule(
    task_id: str, cron: str, destination_url: str, schedule_id: str | None = None
) -> str:
    """Create a recurring QStash schedule that POSTs {"task_id": ...} on `cron`.
    Returns the QStash schedule_id (store it — needed to pause/resume/delete).

    Pass an existing `schedule_id` to update that schedule's cron/body in
    place (used when editing a task) instead of creating a second one."""
    return _get_client().schedule.create_json(
        destination=destination_url,
        cron=cron,
        body={"task_id": task_id},
        method="POST",
        retries=2,
        schedule_id=schedule_id,
    )


def pause_schedule(qstash_schedule_id: str) -> None:
    _get_client().schedule.pause(qstash_schedule_id)


def resume_schedule(qstash_schedule_id: str) -> None:
    _get_client().schedule.resume(qstash_schedule_id)


def delete_schedule(qstash_schedule_id: str) -> None:
    _get_client().schedule.delete(qstash_schedule_id)
