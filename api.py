"""FastAPI service exposing the agentic RAG assistant as a REST microservice.

Run:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agent import Agent
from src import transcribe as stt

app = FastAPI(
    title="Senior Care RAG Agent API",
    version="0.1.0",
    description="Agentic retrieval-augmented generation over a senior-care knowledge base.",
)

FRONTEND = Path(__file__).parent / "frontend" / "index.html"

# Build the agent once at startup (index + embedder loaded/created here).
_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


class AskRequest(BaseModel):
    query: str = Field(..., examples=["Who is eligible for PACE?"])


class Citation(BaseModel):
    n: int
    source: str
    title: str
    score: float
    text: str = ""


class AskResponse(BaseModel):
    query: str
    answer: str
    grounded: bool
    citations: list[Citation]
    trace: list[str]


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the custom chat frontend."""
    return FileResponse(FRONTEND)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    state = get_agent().run(req.query)
    return AskResponse(
        query=req.query,
        answer=state.get("answer", ""),
        grounded=bool(state.get("grounded", False)),
        citations=state.get("citations", []),
        trace=state.get("notes", []),
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., examples=["And for someone under 55?"])
    history: list[ChatMessage] = Field(default_factory=list)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Streaming, multi-turn chat. Emits Server-Sent Events: a `sources` event,
    then many `token` events, then a final `done` event with the grounded flag."""
    agent = get_agent()
    history = [m.model_dump() for m in req.history]

    def event_stream():
        for event in agent.stream(req.query, history):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> dict:
    """Optional server-side speech-to-text (Whisper) for an uploaded audio file.
    Returns {"text": ...}. Requires the audio extra (requirements-audio.txt)."""
    if not stt.available():
        raise HTTPException(
            status_code=501,
            detail="Server-side transcription isn't installed. Run: pip install -r requirements-audio.txt",
        )
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        text = stt.transcribe(path)
    finally:
        os.unlink(path)
    return {"text": text}
