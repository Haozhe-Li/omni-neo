import dotenv

dotenv.load_dotenv()

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.agent import SYSTEM_PROMPTS, initialize_agents
from core.database.postgresql_saver import setup_checkpointer, teardown_checkpointer
from core.prompt_guard import register_sensitive_prompts
from core.database.db_user_threads import setup_thread_search
from core.database.db_user_files import setup_user_files_table
from core.routers import chat, uploads, threads, users, misc


@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_checkpointer()
    setup_user_files_table()
    setup_thread_search()
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
