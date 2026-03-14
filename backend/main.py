"""FastAPI backend with LangGraph + MCP orchestration."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import uuid
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cache import ResponseCache
from config import settings
from database import get_conversation, get_recent_conversation, init_db, save_message
from eval.run import evaluate_dataset
from observability import (
    init_observability,
    metrics_content_type,
    metrics_payload,
)

LOGGER = logging.getLogger("chat.bootstrap")
CACHE = ResponseCache()
AGENT_SERVICE_URL = getattr(settings, "agent_service_url", "http://agent-service:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_observability()
    init_db()
    app.state.http_client = httpx.AsyncClient(
        base_url=AGENT_SERVICE_URL,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    LOGGER.info("API service started, agent service at %s", AGENT_SERVICE_URL)
    yield
    await app.state.http_client.aclose()
    CACHE.close()


app = FastAPI(
    title="Chat API",
    description="Single-agent reflex chat API with MCP tools and LangGraph orchestration",
    version="2.0.0",
    lifespan=lifespan,
)

_CORS_ORIGINS = [
    origin.strip()
    for origin in settings.cors_allowed_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    debug: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    cache_hit: bool
    debug_metadata: Optional[Dict[str, Any]] = None


class MessageItem(BaseModel):
    role: str
    content: str
    created_at: Optional[str] = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageItem]


class EvalRequest(BaseModel):
    dataset_path: str


class EvalResponse(BaseModel):
    run_id: str
    created_at: str
    model_name: str
    enabled_tools: List[str]
    metrics: Dict[str, Any]


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Chat API is running!", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        session_id = request.session_id or str(uuid.uuid4())
        history_limit = settings.chat_history_max_turns

        history = await run_in_threadpool(get_recent_conversation, session_id, history_limit)
        await run_in_threadpool(save_message, session_id, "user", request.message)

        # Check cache first
        cached_response = None if request.debug else CACHE.get(question=request.message)
        if cached_response:
            await run_in_threadpool(save_message, session_id, "assistant", cached_response)
            return ChatResponse(
                response=cached_response,
                session_id=session_id,
                cache_hit=True,
                debug_metadata={"cache_hit": True} if request.debug else None,
            )

        # Call agent service via HTTP
        client: httpx.AsyncClient = app.state.http_client
        agent_response = await client.post(
            "/invoke",
            json={
                "user_message": request.message,
                "history": history,
                "session_id": session_id,
            },
        )
        agent_response.raise_for_status()
        agent_data = agent_response.json()

        response = agent_data.get("final_response", "I could not generate a response.")

        # Cache if no fallback was used
        if not agent_data.get("fallback_used", False):
            CACHE.set(question=request.message, response=response)

        await run_in_threadpool(save_message, session_id, "assistant", response)

        debug_data = None
        if request.debug:
            debug_data = {
                "selected_tool": agent_data.get("selected_tool"),
                "retries": agent_data.get("retry_count", 0),
                "fallback_used": agent_data.get("fallback_used", False),
                "validation_passed": agent_data.get("validation_passed", False),
                "cache_hit": False,
            }

        return ChatResponse(
            response=response,
            session_id=session_id,
            cache_hit=False,
            debug_metadata=debug_data,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Agent service error: {exc}") from exc
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail="Agent service unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"An error occurred: {exc}") from exc


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str) -> HistoryResponse:
    messages = await run_in_threadpool(get_conversation, session_id)
    items = [
        MessageItem(
            role=msg["role"],
            content=msg["content"],
            created_at=msg.get("created_at"),
        )
        for msg in messages
    ]
    return HistoryResponse(session_id=session_id, messages=items)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}



@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=metrics_payload(), media_type=metrics_content_type())


@app.post("/eval/run", response_model=EvalResponse)
async def run_eval(request: EvalRequest) -> EvalResponse:
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=400, detail="dataset_path does not exist")
    metrics = await evaluate_dataset(request.dataset_path)
    enabled_tools = [t.strip() for t in settings.mcp_enabled_tools.split(",") if t.strip()]
    return EvalResponse(
        run_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        model_name=settings.litellm_model_name,
        enabled_tools=enabled_tools,
        metrics=metrics,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
