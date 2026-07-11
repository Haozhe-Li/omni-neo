import asyncio

from fastapi import APIRouter, Depends

from core.auth import get_current_user
from core.database.db_user_memories import get_user_memory, delete_user_memory

router = APIRouter(prefix="/api", tags=["memories"])


@router.get("/memories")
async def api_get_memory(user_id: str = Depends(get_current_user)):
    content = await asyncio.to_thread(get_user_memory, user_id)
    return {"content": content}


@router.delete("/memories")
async def api_clear_memory(user_id: str = Depends(get_current_user)):
    await asyncio.to_thread(delete_user_memory, user_id)
    return {"status": "cleared"}
