"""Standalone Agent microservice exposing LangGraph orchestration over HTTP.

This service is deployed separately from the main API service in Kubernetes,
allowing independent scaling of the orchestration/LLM workload.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import ORCHESTRATOR, close_external_connections
from config import settings
from observability import init_observability

LOGGER = logging.getLogger("agent.service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_observability()
    LOGGER.info("Agent service started")
    yield
    close_external_connections()
    LOGGER.info("Agent service stopped")


app = FastAPI(
    title="Agent Service",
    description="LangGraph orchestration microservice",
    version="2.0.0",
    lifespan=lifespan,
)


class InvokeRequest(BaseModel):
    user_message: str
    history: List[Dict[str, Any]] = []
    session_id: str = ""


class InvokeResponse(BaseModel):
    final_response: str
    selected_tool: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False
    validation_passed: bool = False


@app.post("/invoke", response_model=InvokeResponse)
async def invoke(request: InvokeRequest) -> InvokeResponse:
    try:
        state = await ORCHESTRATOR.run(
            user_message=request.user_message,
            history=request.history,
            session_id=request.session_id,
        )
        response = state.get("final_response") or "I could not generate a response."
        fallback_report = state.get("fallback_report") or {}

        return InvokeResponse(
            final_response=response,
            selected_tool=state.get("selected_tool"),
            retry_count=state.get("retry_count", 0),
            fallback_used=fallback_report.get("used", False),
            validation_passed=state.get("validation_passed", False),
        )
    except Exception as exc:
        LOGGER.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "agent"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
