"""Scheduled research tasks: "run this prompt on a schedule, email me the
report" — the backend half. QStash fires a cron webhook -> a fresh chat
thread runs the scheduled agent variant (core/scheduled_agent.py) -> the
report is stored privately on its run row (core/database/db_scheduled_tasks.py)
-> a plain-text summary + full report is emailed via Resend
(core/utils/resend_client.py), linking to the auth-gated /schedule/{run_id}
page on the frontend. Unlike a manually-published Pages report, a scheduled
report is never public by default — only the task's own user can read it
(enforced in api_get_run below); sharing it out to Pages is an explicit,
separate action the user takes from that page.

Every task fire gets its own thread_id (core/scheduled_agent.py runs it as an
ordinary first turn in that thread), so the user can open the resulting
thread afterward and keep chatting with it like any other conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core.auth import get_current_user
from core.database.db_scheduled_tasks import (
    MAX_ACTIVE_TASKS,
    count_active_tasks,
    create_task,
    create_run,
    get_run,
    get_task,
    list_tasks_for_user,
    list_runs_for_task,
    update_task,
    update_task_status,
    update_run,
)
from core.database.db_threads_control import upsert_thread
from core.database.db_user_threads import register_thread
from core.database.db_user_usage import charge_credits
from core.scheduled_agent import run_scheduled_task, ScheduledRunError
from core.scheduled_task_parser import parse_schedule_prompt
from core.utils.qstash_client import (
    create_schedule,
    pause_schedule,
    resume_schedule,
    delete_schedule,
    verify_webhook_signature,
)
from core.utils.resend_client import send_report_email, send_task_confirmation_email

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scheduled_tasks"])

SITE_URL = os.getenv("SITE_URL", "https://omniknows.xyz")
# This backend's own publicly reachable base URL — QStash needs a real URL to
# call back into. Unset in local dev, where there's no public ingress; task
# creation still succeeds (qstash_schedule_id stays None) so the rest of the
# pipeline can be exercised by invoking _execute_run directly.
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "").rstrip("/")


class CreateTaskRequest(BaseModel):
    name: str
    prompt: str
    cron_schedule: str
    email: str
    # Human-readable schedule description (e.g. "Daily at 9:00 AM"), computed
    # client-side from the user's local timezone — the backend only ever sees
    # the UTC cron string, which isn't reconstructable back to local time
    # without knowing the browser's offset. Used solely for the confirmation
    # email's copy; falls back to the raw cron string if omitted.
    schedule_label: str = ""


class EditTaskRequest(BaseModel):
    name: str
    prompt: str
    cron_schedule: str


class UpdateTaskRequest(BaseModel):
    action: str  # "pause" | "resume" | "delete"


class ParsePromptRequest(BaseModel):
    text: str


@router.post("/schedule_task/parse")
def api_parse_schedule_prompt(request: ParsePromptRequest, user_id: str = Depends(get_current_user)):
    """Turn a casual request (the Settings quick-create box) into
    {title, instruction, schedule_time} for the frontend to load into the
    create form. Requires auth purely to keep this LLM call gated like every
    other endpoint here — it doesn't touch the caller's data."""
    try:
        parsed = parse_schedule_prompt(request.text)
    except Exception as exc:
        logger.error(f"[scheduled_tasks] parse_schedule_prompt failed: {exc}")
        raise HTTPException(status_code=502, detail="Failed to understand that request.")
    return parsed.model_dump()


@router.post("/schedule_task")
def api_create_task(request: CreateTaskRequest, user_id: str = Depends(get_current_user)):
    if count_active_tasks(user_id) >= MAX_ACTIVE_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"You can have at most {MAX_ACTIVE_TASKS} scheduled tasks at a time.",
        )

    task_id = uuid.uuid4().hex

    qstash_schedule_id = None
    if BACKEND_PUBLIC_URL:
        try:
            qstash_schedule_id = create_schedule(
                task_id, request.cron_schedule, f"{BACKEND_PUBLIC_URL}/scheduled_task/run"
            )
        except Exception as exc:
            logger.error(f"[scheduled_tasks] QStash schedule creation failed: {exc}")
            raise HTTPException(status_code=502, detail="Failed to create the schedule.")
    else:
        logger.warning(
            "[scheduled_tasks] BACKEND_PUBLIC_URL not set — creating task without a live QStash schedule."
        )

    ok = create_task(
        task_id, user_id, request.name, request.email, request.prompt, request.cron_schedule, qstash_schedule_id
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create task.")

    # Best-effort, matching send_report_email's contract — the task is
    # already committed, so a flaky email send shouldn't turn into a 500 for
    # something the user will see succeeded in their task list regardless.
    send_task_confirmation_email(
        request.email,
        request.name,
        request.schedule_label or request.cron_schedule,
        f"{SITE_URL}/settings/scheduled-research",
    )

    return {"task_id": task_id, "qstash_schedule_id": qstash_schedule_id}


@router.get("/schedule_task")
def api_list_tasks(user_id: str = Depends(get_current_user)):
    tasks = list_tasks_for_user(user_id)
    for t in tasks:
        t["runs"] = list_runs_for_task(t["task_id"])
    return {"tasks": tasks}


@router.get("/schedule_task/{task_id}")
def api_get_task(task_id: str, user_id: str = Depends(get_current_user)):
    task = get_task(task_id)
    if not task or task["user_id"] != user_id or task["status"] == "deleted":
        raise HTTPException(status_code=404, detail="Task not found.")
    task["runs"] = list_runs_for_task(task_id)
    return task


@router.get("/schedule_task/run/{run_id}")
def api_get_run(run_id: str, user_id: str = Depends(get_current_user)):
    """Private report view — the target of a scheduled report's email link
    and the frontend's /schedule/{run_id} page. Ownership is checked through
    the run's parent task, not stored redundantly on the run row itself; a
    404 (not 403) either way avoids confirming a run_id exists to a
    non-owner probing IDs."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Report not found.")
    task = get_task(run["task_id"])
    if not task or task["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Report not found.")
    return {
        "run_id": run["run_id"],
        "task_id": run["task_id"],
        "task_name": task["name"],
        "status": run["status"],
        "error": run["error"],
        "title": run["title"],
        "report": run["report_markdown"],
        "sources": run["sources"] or [],
        "summary": run["summary"],
        "created_at": run["created_at"],
    }


@router.put("/schedule_task/{task_id}")
def api_edit_task(task_id: str, request: EditTaskRequest, user_id: str = Depends(get_current_user)):
    task = get_task(task_id)
    if not task or task["user_id"] != user_id or task["status"] == "deleted":
        raise HTTPException(status_code=404, detail="Task not found.")

    qstash_schedule_id = task.get("qstash_schedule_id")
    if BACKEND_PUBLIC_URL:
        try:
            # Passing schedule_id updates the existing QStash schedule in place
            # (new cron + body) instead of creating a second one.
            qstash_schedule_id = create_schedule(
                task_id, request.cron_schedule, f"{BACKEND_PUBLIC_URL}/scheduled_task/run",
                schedule_id=qstash_schedule_id,
            )
        except Exception as exc:
            logger.error(f"[scheduled_tasks] QStash update failed for {task_id}: {exc}")
            raise HTTPException(status_code=502, detail="Failed to update the schedule.")

    ok = update_task(task_id, user_id, name=request.name, prompt=request.prompt, cron_schedule=request.cron_schedule)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update task.")

    return {"status": "ok", "qstash_schedule_id": qstash_schedule_id}


@router.patch("/schedule_task/{task_id}")
def api_update_task(task_id: str, request: UpdateTaskRequest, user_id: str = Depends(get_current_user)):
    task = get_task(task_id)
    if not task or task["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Task not found.")

    qstash_schedule_id = task.get("qstash_schedule_id")
    try:
        if request.action == "pause":
            if qstash_schedule_id:
                pause_schedule(qstash_schedule_id)
            update_task_status(task_id, user_id, "paused")
        elif request.action == "resume":
            if qstash_schedule_id:
                resume_schedule(qstash_schedule_id)
            update_task_status(task_id, user_id, "active")
        elif request.action == "delete":
            if qstash_schedule_id:
                delete_schedule(qstash_schedule_id)
            update_task_status(task_id, user_id, "deleted")
        else:
            raise HTTPException(status_code=400, detail="Unknown action.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[scheduled_tasks] QStash {request.action} failed for {task_id}: {exc}")
        raise HTTPException(status_code=502, detail=f"Failed to {request.action} the schedule.")

    return {"status": "ok"}


async def _execute_run(run_id: str, task_id: str, thread_id: str, user_id: str, email: str, prompt: str) -> None:
    """The actual work of one task firing — awaited inside the webhook request
    itself (see api_run_task for why it must not be fire-and-forget)."""
    update_run(run_id, status="running")

    # Charged the same as an interactive turn — a scheduled run is not a
    # free ride around the credit system. Checked/charged up front, before
    # the (expensive) agent call, mirroring _charge_or_429 in chat.py: if
    # the user is already over their cap, skip the run entirely rather than
    # burning tokens on a report that can't be billed.
    usage = await asyncio.to_thread(charge_credits, user_id, "scheduled")
    if not usage["charged"]:
        logger.warning(f"[scheduled_tasks] run {run_id} skipped — usage limit exceeded for {user_id}")
        update_run(run_id, status="failed", error="Usage limit exceeded — this run was skipped.")
        return

    try:
        result = await run_scheduled_task(thread_id, user_id, prompt)
        # Report stays private to this run (see the module docstring in
        # core/database/db_scheduled_tasks.py) — the emailed link opens the
        # auth-gated /schedule/{run_id} page, not a public Pages URL. Users
        # can explicitly copy a report out to Pages via that page's Share button.
        report_url = f"{SITE_URL}/schedule/{run_id}"
        await asyncio.to_thread(
            send_report_email, email, result["title"], result["summary"], result["report"], report_url
        )
        update_run(
            run_id,
            status="success",
            title=result["title"],
            report_markdown=result["report"],
            sources=result["sources"],
            summary=result["summary"],
        )
    except ScheduledRunError as exc:
        logger.error(f"[scheduled_tasks] run {run_id} failed: {exc}")
        update_run(run_id, status="failed", error=str(exc))
    except Exception as exc:
        traceback.print_exc()
        update_run(run_id, status="failed", error=str(exc))


@router.post("/scheduled_task/run")
async def api_run_task(request: Request):
    """QStash webhook target — verifies the signature, then runs the whole
    task INSIDE the request and only responds when it's done.

    Deliberately not fire-and-forget: Cloud Run only allocates CPU while a
    request is in flight, so a detached asyncio task runs on a throttled
    (~zero) CPU — tool calls that take 1-2s in interactive chat ballooned to
    15-30s and Redis connects timed out outright. Holding the request open
    keeps full CPU for the run's duration. Budget-wise this fits: QStash
    waits up to the schedule's 600s timeout (qstash_client.create_schedule),
    matching Cloud Run's 600s request timeout, and a run normally finishes
    in well under that."""
    body_bytes = await request.body()
    body_str = body_bytes.decode()
    signature = request.headers.get("upstash-signature", "")

    verify_url = f"{BACKEND_PUBLIC_URL}/scheduled_task/run" if BACKEND_PUBLIC_URL else None
    if not verify_webhook_signature(signature=signature, body=body_str, url=verify_url):
        raise HTTPException(status_code=401, detail="Invalid QStash signature.")

    payload = json.loads(body_str)
    task_id = payload.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="Missing task_id.")

    task = get_task(task_id)
    if not task or task["status"] != "active":
        # Not an error from QStash's perspective — just nothing to do (task
        # was paused/deleted after this delivery was already queued).
        return {"status": "skipped"}

    message_id = request.headers.get("upstash-message-id")
    run_id = uuid.uuid4().hex
    if not create_run(run_id, task_id, qstash_message_id=message_id):
        # Same message_id already has a run row — this is a QStash retry of a
        # delivery we already started; do not run it twice.
        return {"status": "duplicate"}

    thread_id = uuid.uuid4().hex
    upsert_thread(thread_id, task["user_id"])
    register_thread(thread_id, task["user_id"], origin="scheduled_task")

    await _execute_run(run_id, task_id, thread_id, task["user_id"], task["email"], task["prompt"])

    return {"status": "completed", "run_id": run_id}
