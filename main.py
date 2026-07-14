import dotenv

dotenv.load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.agent import SYSTEM_PROMPTS, initialize_agents
from core.database.checkpointer import setup_checkpointer, teardown_checkpointer
from core.prompt_guard import register_sensitive_prompts
from core.routers import chat, uploads, threads, users, misc, memories, scheduled_tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Table schema is now managed directly in Supabase (see schema.sql) — DDL
    # can't run over the PostgREST HTTP API, so there are no setup_*_table()
    # calls here anymore. The checkpointer is just an Upstash REST client, so
    # setup is instant (no pool to open).
    await setup_checkpointer()
    initialize_agents()
    yield
    await teardown_checkpointer()


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
