"""Agent service entrypoints backed by LangGraph orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from cache import ResponseCache
from config import settings
from observability import get_logger
from orchestrator.engine import OrchestratorGraph
from tools.adapters import TavilyDirectAdapter
from tools.registry import ToolRegistry

CACHE = ResponseCache()
LOGGER = get_logger()


def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=settings.litellm_api_key,
        openai_api_base=settings.litellm_base_url,
        model_name=settings.litellm_model_name,
        temperature=settings.litellm_temperature,
        request_timeout=settings.litellm_request_timeout,
        max_retries=settings.litellm_max_retries,
        max_tokens=settings.litellm_max_tokens,
    )


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    tavily_timeout_s = min(max(settings.mcp_tool_timeout_ms / 1000.0, 4.0), 10.0)
    if settings.tavily_timeout_seconds:
        tavily_timeout_s = settings.tavily_timeout_seconds

    tavily_timeout_ms = int(tavily_timeout_s * 1000)

    registry.register(
        tool_id="tavily_search",
        adapter=TavilyDirectAdapter(
            api_key=settings.tavily_api_key,
            timeout_seconds=tavily_timeout_s,
            topic=settings.tavily_topic,
            search_depth=settings.tavily_search_depth,
            max_results=settings.tavily_max_results,
        ),
        enabled=True,
        priority=10,
        timeout_ms=tavily_timeout_ms,
        max_retries=settings.tavily_max_retries,
        initial_backoff_ms=settings.mcp_tool_initial_backoff_ms,
    )
    return registry


ORCHESTRATOR = OrchestratorGraph(registry=_build_registry(), llm=_create_llm(), logger=LOGGER)


async def get_chat_response_with_cache_info(
    user_message: str,
    history: Optional[List[Dict]] = None,
    session_id: str = "",
    debug: bool = False,
) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
    """Generate response through orchestration graph with cache support."""
    cached_response = None if debug else CACHE.get(question=user_message)
    if cached_response:
        debug_data = (
            {
                "selected_tool": None,
                "retries": 0,
                "fallback_used": False,
                "validation_passed": True,
                "cache_hit": True,
            }
            if debug
            else None
        )
        return cached_response, True, debug_data

    state = await ORCHESTRATOR.run(
        user_message=user_message,
        history=history or [],
        session_id=session_id,
    )
    response = state.get("final_response") or "I could not generate a response."

    fallback_report = state.get("fallback_report") or {}
    if not fallback_report.get("used", False):
        CACHE.set(question=user_message, response=response)

    debug_data = None
    if debug:
        debug_data = {
            "selected_tool": state.get("selected_tool"),
            "retries": state.get("retry_count", 0),
            "fallback_used": fallback_report.get("used", False),
            "validation_passed": state.get("validation_passed", False),
            "cache_hit": False,
        }

    return response, False, debug_data


def close_external_connections() -> None:
    """Close pooled connections used by external services."""
    CACHE.close()
