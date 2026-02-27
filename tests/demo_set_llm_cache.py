
import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

# 保存原函数
_original_generate_with_cache = BaseChatModel._generate_with_cache
_original_agenerate_with_cache = BaseChatModel._agenerate_with_cache

def _generate_with_cache_fixed(self, messages, *args, **kwargs):
    print("DEBUG: Sync cache lookup, stripping IDs")
    messages_clean = [msg.copy(update={"id": None}) for msg in messages]
    return _original_generate_with_cache(self, messages_clean, *args, **kwargs)

async def _agenerate_with_cache_fixed(self, messages, *args, **kwargs):
    print("DEBUG: ASYNC cache lookup, stripping IDs")
    messages_clean = [msg.copy(update={"id": None}) for msg in messages]
    return await _original_agenerate_with_cache(self, messages_clean, *args, **kwargs)

# 覆盖两个
BaseChatModel._generate_with_cache = _generate_with_cache_fixed
BaseChatModel._agenerate_with_cache = _agenerate_with_cache_fixed

load_dotenv()

from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache

import redis

r = redis.Redis(
    host='redis-16666.c280.us-central1-2.gce.cloud.redislabs.com',
    port=16666,
    decode_responses=True,
    username="default",
    password="1wwaw5HlXSaZsavEoDwaF10OUpOU6srW",
)

set_llm_cache(RedisCache(redis_=r))
llm = init_chat_model("google_genai:gemini-3-flash-preview")

test_agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant that provides concise answers.",
    name="test_agent",
)


# Function to measure execution time
def timed_completion(prompt):
    start_time = time.time()
    message = {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }
    result = test_agent.invoke(message)
    end_time = time.time()
    return result, end_time - start_time


# First call (not cached)
prompt = "Explain the concept of caching in three sentences."
result1, time1 = timed_completion(prompt)
print(f"First call (not cached):\nTime: {time1:.2f} seconds\n")

# Second call (should be cached)
result2, time2 = timed_completion(prompt)
print(f"Second call (cached):\nTime: {time2:.2f} seconds\n")

print(f"Speed improvement: {time1 / time2:.2f}x faster")

print("Cache cleared")