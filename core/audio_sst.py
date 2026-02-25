
import os
from groq import AsyncGroq

api_key = os.getenv("GROQ_API_KEY")
client = AsyncGroq(api_key=api_key) if api_key else None


async def get_text_from_audio(file: bytes) -> str:
  if client is None:
    raise RuntimeError("GROQ_API_KEY is not configured on the server.")

  transcription = await client.audio.transcriptions.create(
    file=("audio.m4a", file),
    model="whisper-large-v3-turbo",
    temperature=0,
    response_format="verbose_json",
  )
  return transcription.text