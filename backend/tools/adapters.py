from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import redis
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config import settings
from contracts import ToolResult


class McpToolAdapter(ABC):
    """Abstract contract for tool adapters."""

    @abstractmethod
    async def execute(self, tool_input: Dict[str, Any]) -> ToolResult:
        """Execute tool with structured input."""


class _RedisCache:
    """Lazy Redis client shared by tool adapters."""

    def __init__(self) -> None:
        self._client: Any = None  # None = uninitialised, False = unavailable

    def get_client(self) -> Optional[redis.Redis]:
        if self._client is False:
            return None
        if self._client is None:
            try:
                if not settings.enable_response_cache:
                    self._client = False
                    return None
                r = redis.from_url(settings.redis_url, decode_responses=True)
                r.ping()
                self._client = r
            except Exception:
                self._client = False
        return self._client


_tool_redis = _RedisCache()


def _normalize_query(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def _tavily_cache_key(query: str, topic: str, depth: str, max_n: int) -> str:
    norm = _normalize_query(query)
    h = hashlib.sha256(f"tavily:{topic}:{depth}:{max_n}:{norm}".encode()).hexdigest()
    return f"tavily_result:v1:{h}"


class TavilyDirectAdapter(McpToolAdapter):
    """Tavily integration via Python SDK - low latency, no npx subprocess."""

    def __init__(
        self,
        api_key: Optional[str],
        timeout_seconds: float = 15.0,
        topic: str = "general",
        search_depth: str = "basic",
        max_results: int = 3,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.topic = topic
        self.search_depth = search_depth
        self.max_results = max_results
        self._static_readiness_error: Optional[str] = None

    def readiness_error(self) -> Optional[str]:
        if self._static_readiness_error is None:
            if not self.api_key:
                self._static_readiness_error = (
                    "Tavily API key is not configured. Set TAVILY_API_KEY."
                )
            elif importlib.util.find_spec("tavily") is None:
                self._static_readiness_error = (
                    "tavily-python package is missing in backend runtime."
                )
            else:
                self._static_readiness_error = ""
        return self._static_readiness_error or None

    def to_langchain_tool(self) -> StructuredTool:
        class TavilyInput(BaseModel):
            query: str = Field(description="Web search query for current events, news, prices, or live data")

        return StructuredTool(
            name="tavily_search",
            description=(
                "Search the web for current information, news, live data, prices, or recent events. "
                "Use when the user needs up-to-date information that is not in the knowledge base."
            ),
            args_schema=TavilyInput,
            coroutine=self.execute,
        )

    async def execute(self, tool_input: Dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        query = (tool_input.get("query") or "").strip()

        if not self.api_key:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error="Tavily API key is not configured.",
                error_code="not_configured",
                is_transient_error=False,
            )
        if not query:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error="Search query is empty.",
                error_code="invalid_input",
                is_transient_error=False,
            )

        redis_client = _tool_redis.get_client()
        if redis_client:
            key = _tavily_cache_key(query, self.topic, self.search_depth, self.max_results)
            try:
                raw = redis_client.get(key)
                if raw:
                    return ToolResult(
                        tool_id="tavily_search",
                        success=True,
                        output=json.loads(raw),
                        latency_ms=int((time.perf_counter() - start) * 1000),
                    )
            except Exception:
                pass

        try:
            from tavily import AsyncTavilyClient
        except ImportError as exc:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error=f"tavily-python not installed: pip install tavily-python. {exc}",
                error_code="sdk_missing",
                is_transient_error=False,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        try:
            async with AsyncTavilyClient(api_key=self.api_key) as client:
                response = await asyncio.wait_for(
                    client.search(
                        query=query,
                        search_depth=self.search_depth,
                        topic=self.topic,
                        max_results=self.max_results,
                        include_answer=False,
                        include_raw_content=False,
                        include_images=False,
                        timeout=self.timeout_seconds,
                    ),
                    timeout=self.timeout_seconds,
                )
        except TimeoutError:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error="Tavily search timed out.",
                error_code="timeout",
                is_transient_error=True,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error=f"Tavily search failed: {exc}",
                error_code="api_error",
                is_transient_error=True,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        results = getattr(response, "results", None) or (
            response.get("results", []) if isinstance(response, dict) else []
        )
        parsed_content = []
        for r in (results or []):
            c = getattr(r, "content", None) if not isinstance(r, dict) else r.get("content", "")
            if c:
                parsed_content.append(str(c))

        output = {"query": query, "content": parsed_content}

        if redis_client:
            key = _tavily_cache_key(query, self.topic, self.search_depth, self.max_results)
            try:
                redis_client.setex(key, settings.redis_cache_ttl_seconds, json.dumps(output, ensure_ascii=False))
            except Exception:
                pass

        return ToolResult(
            tool_id="tavily_search",
            success=True,
            output=output,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


class TavilyMcpAdapter(McpToolAdapter):
    """Tavily integration through the official MCP server (slower - uses npx subprocess)."""

    def __init__(
        self,
        api_key: Optional[str],
        timeout_seconds: float = 20.0,
        topic: str = "news",
        search_depth: str = "advanced",
        max_results: int = 5,
        package: str = "tavily-mcp@0.1.3",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.topic = topic
        self.search_depth = search_depth
        self.max_results = max_results
        self.package = package

    async def execute(self, tool_input: Dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        query = (tool_input.get("query") or "").strip()

        if not self.api_key:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error="Tavily API key is not configured.",
                error_code="not_configured",
                is_transient_error=False,
            )
        if not query:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error="Search query is empty.",
                error_code="invalid_input",
                is_transient_error=False,
            )

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error=f"MCP Python SDK is not available: {exc}",
                error_code="mcp_sdk_missing",
                is_transient_error=False,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", self.package],
            env={
                **os.environ,
                "TAVILY_API_KEY": self.api_key,
                "DEFAULT_PARAMETERS": (
                    "{"
                    f"\"topic\":\"{self.topic}\","
                    f"\"search_depth\":\"{self.search_depth}\","
                    f"\"max_results\":{self.max_results},"
                    "\"include_images\":false,"
                    "\"include_raw_content\":false"
                    "}"
                ),
            },
        )

        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool("tavily-search", {"query": query})
        except TimeoutError:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error="Tavily MCP request timed out.",
                error_code="timeout",
                is_transient_error=True,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        except FileNotFoundError:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error="npx is not available. Install Node.js 20+ to run Tavily MCP.",
                error_code="npx_missing",
                is_transient_error=False,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        except Exception as exc:
            return ToolResult(
                tool_id="tavily_search",
                success=False,
                error=f"Tavily MCP request failed: {exc}",
                error_code="mcp_error",
                is_transient_error=True,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        parsed_content = []
        for item in getattr(result, "content", []):
            if hasattr(item, "text"):
                parsed_content.append(item.text)
            else:
                parsed_content.append(str(item))

        return ToolResult(
            tool_id="tavily_search",
            success=not getattr(result, "isError", False),
            output={"query": query, "content": parsed_content},
            error="\n".join(parsed_content) if getattr(result, "isError", False) else None,
            error_code="tool_error" if getattr(result, "isError", False) else None,
            is_transient_error=False,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
