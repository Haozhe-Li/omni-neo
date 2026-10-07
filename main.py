import dotenv

dotenv.load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.agent import SYSTEM_PROMPTS, initialize_agents
from core.database import pg
from core.database.checkpointer import setup_checkpointer, teardown_checkpointer
from core.intent_router import warm_intent_router
from core.prompt_guard import register_sensitive_prompts
from core.utils.redis_client import close_async_redis
from core.voice.agent import initialize_voice_agent
from core.routers import chat, uploads, threads, users, misc, memories, scheduled_tasks, evals, voice, feedback, shares, collector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the Postgres pools now so a bad DATABASE_URL fails the deploy instead of
    # the first user request. Schema is applied separately (python -m scripts.init_db).
    await pg.apool()
    pg.pool()
    await setup_checkpointer()
    initialize_agents()
    initialize_voice_agent()
    # Build the context-enrichment intent router now, so no request pays the 1-2s.
    # Bounded and non-fatal: if the embedding service is down the app still starts
    # and every turn uses the LLM scout until the router builds (core/intent_router.py).
    await warm_intent_router()
    yield
    await teardown_checkpointer()
    await close_async_redis()
    await pg.aclose_pools()


app = FastAPI(title="Omni Agent API", lifespan=lifespan)

register_sensitive_prompts(SYSTEM_PROMPTS)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(uploads.router)
app.include_router(threads.router)
app.include_router(users.router)
app.include_router(misc.router)
app.include_router(memories.router)
app.include_router(scheduled_tasks.router)
app.include_router(evals.router)
app.include_router(voice.router)
app.include_router(feedback.router)
app.include_router(shares.router)
app.include_router(collector.router)
