"""FastAPI web chat: conversation storage, chat API, and streaming TTS.

Model loading, Best-of-N selection, RAG, and the calculator flow live in engine.py.
"""

import asyncio
import base64
import importlib.metadata
import json
import os
import re
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import TTSConfig
from engine import ChatEngine, utc_now
from tools.tts import VOICES as TTS_VOICES, get_tts, list_voices as tts_list_voices, warm_up as tts_warm_up

TTS_CONFIG = TTSConfig()


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ConversationStore:
    """Small local JSON store for conversations and turn metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"version": 1, "conversations": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded.get("conversations"), dict):
                    self.data = loaded
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: Could not load conversation history: {exc}")

    def _save(self):
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)

    def create(self):
        now = utc_now()
        conversation = {
            "id": str(uuid.uuid4()),
            "title": "New chat",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        with self.lock:
            self.data["conversations"][conversation["id"]] = conversation
            self._save()
        return conversation

    def get(self, conversation_id: str):
        with self.lock:
            conversation = self.data["conversations"].get(conversation_id)
            return json.loads(json.dumps(conversation)) if conversation else None

    def delete(self, conversation_id: str):
        with self.lock:
            if self.data["conversations"].pop(conversation_id, None) is None:
                return False
            self._save()
            return True

    def list(self):
        with self.lock:
            conversations = list(self.data["conversations"].values())
            return [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "message_count": len(item["messages"]),
                }
                for item in sorted(
                    conversations, key=lambda item: item["updated_at"], reverse=True
                )
            ]

    def append_turn(self, conversation_id: str, user_text: str, assistant_text: str, metadata: dict):
        now = utc_now()
        with self.lock:
            conversation = self.data["conversations"].get(conversation_id)
            if conversation is None:
                return None
            if not conversation["messages"]:
                title = re.sub(r"\s+", " ", user_text).strip()
                conversation["title"] = title[:58] + ("…" if len(title) > 58 else "")
            conversation["messages"].extend([
                {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": user_text,
                    "timestamp_utc": now,
                },
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": assistant_text,
                    "timestamp_utc": now,
                    "metadata": metadata,
                },
            ])
            conversation["updated_at"] = now
            self._save()
            return json.loads(json.dumps(conversation))


engine: Optional[ChatEngine] = None
conversation_store = ConversationStore(Path(__file__).resolve().parent / "conversation_history.json")

# Bind to localhost by default so other machines on the network cannot read or
# delete conversations. Set LOCALQWEN_HOST=0.0.0.0 to expose it deliberately.
HOST = os.environ.get("LOCALQWEN_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOCALQWEN_PORT", "8000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Loading model on startup...")
    try:
        kokoro_version = importlib.metadata.version("kokoro-onnx")
        print(f"kokoro-onnx version: {kokoro_version}")
    except importlib.metadata.PackageNotFoundError:
        print("kokoro-onnx version: not installed")

    # TTS (download if missing + load) runs alongside the LLM and RAG loading.
    # The server does not wait for it: a /tts request that arrives first simply
    # waits for the same load to finish.
    tts_task = asyncio.create_task(asyncio.to_thread(tts_warm_up)) if TTS_CONFIG.enabled else None
    engine = await asyncio.to_thread(ChatEngine, None, TTS_CONFIG.enabled)
    yield
    if tts_task is not None and not tts_task.done():
        tts_task.cancel()  # the worker thread finishes on its own; this only drops the handle
    print("Shutting down...")


app = FastAPI(title="Local Qwen Chat API", lifespan=lifespan)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
# No CORS middleware: the UI is served from this same origin, so cross-origin
# access is not needed and stays blocked by the browser.


@app.get("/conversations")
def list_conversations():
    return {"conversations": conversation_store.list()}


@app.post("/conversations")
def create_conversation():
    return conversation_store.create()


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conversation = conversation_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    if not conversation_store.delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "conversation_id": conversation_id}


# Plain `def` endpoints run in FastAPI's thread pool, so a long generation no
# longer blocks the event loop (TTS streaming and /health keep working).
# ChatEngine.respond serialises GPU work with its own lock.
@app.post("/chat")
def chat(request: ChatRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    conversation = (
        conversation_store.get(request.conversation_id)
        if request.conversation_id else conversation_store.create()
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    history = []
    messages = conversation["messages"]
    for index in range(0, len(messages) - 1, 2):
        user_message, assistant_message = messages[index:index + 2]
        if user_message["role"] == "user" and assistant_message["role"] == "assistant":
            history.append((user_message["content"], assistant_message["content"]))

    try:
        response, metadata = engine.respond(request.message, history)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    conversation_store.append_turn(conversation["id"], request.message, response, metadata)
    return {
        "conversation_id": conversation["id"],
        "response": response,
        "metadata": metadata,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.get("/tts/voices")
def tts_voices():
    """Return the available Kokoro TTS voice catalogue."""
    return {"voices": tts_list_voices()}


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"


def _tts_response(text: str, voice: str) -> StreamingResponse:
    """
    Stream synthesised speech as Server-Sent Events.

    Each SSE ``data:`` event carries a base64-encoded WAV blob for one sentence
    chunk (24 kHz, 16-bit PCM mono). A final ``event: done`` signals completion.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    if voice not in TTS_VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")
    tts = get_tts()

    async def generate():
        try:
            async for samples, sample_rate in tts.synthesize_stream(text, voice=voice):
                wav_bytes = tts.chunk_to_wav_bytes(samples, sample_rate)
                b64 = base64.b64encode(wav_bytes).decode("ascii")
                yield f"data: {b64}\n\n"
        except Exception as exc:
            message = str(exc).replace("\n", " ")
            yield f"event: error\ndata: {message}\n\n"
        finally:
            yield "event: done\ndata: end\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/tts/stream")
async def tts_stream_post(request: TTSRequest):
    """Preferred TTS endpoint: text goes in the JSON body, so long answers are fine."""
    return _tts_response(request.text, request.voice)


@app.get("/tts/stream")
async def tts_stream_get(
    text: str = Query(..., description="Text to synthesise"),
    voice: str = Query("af_heart", description="Kokoro voice id"),
):
    """GET variant kept for short texts and manual testing (limited by URL length)."""
    return _tts_response(text, voice)


@app.get("/")
def root():
    return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
