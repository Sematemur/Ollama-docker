"""Tool registry with retry-aware execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from contracts import ToolResult
from tools.adapters import McpToolAdapter


@dataclass
class ToolRegistration:
    """Registered tool metadata."""

    tool_id: str
    adapter: McpToolAdapter
    enabled: bool = True
    priority: int = 100
    timeout_ms: int = 5000
    max_retries: int = 2
    initial_backoff_ms: int = 200


class ToolRegistry:
    """In-memory registry for managing MCP tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolRegistration] = {}

    def register(
        self,
        tool_id: str,
        adapter: McpToolAdapter,
        enabled: bool = True,
        priority: int = 100,
        timeout_ms: int = 5000,
        max_retries: int = 2,
        initial_backoff_ms: int = 200,
    ) -> None:
        self._tools[tool_id] = ToolRegistration(
            tool_id=tool_id,
            adapter=adapter,
            enabled=enabled,
            priority=priority,
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            initial_backoff_ms=initial_backoff_ms,
        )

    def enable(self, tool_id: str, enabled: bool) -> None:
        if tool_id in self._tools:
            self._tools[tool_id].enabled = enabled

    def resolve_available_tools(self) -> List[str]:
        enabled = [t for t in self._tools.values() if t.enabled]
        enabled.sort(key=lambda t: t.priority)
        return [t.tool_id for t in enabled]

    def get(self, tool_id: str) -> Optional[ToolRegistration]:
        return self._tools.get(tool_id)

    def list_tools(self) -> List[ToolRegistration]:
        tools = list(self._tools.values())
        tools.sort(key=lambda t: t.priority)
        return tools

    async def execute(self, tool_id: str, tool_input: Dict[str, Any]) -> ToolResult:
        registration = self._tools.get(tool_id)
        if not registration:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                error=f"Unknown tool: {tool_id}",
                error_code="tool_not_found",
                is_transient_error=False,
            )
        if not registration.enabled:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                error=f"Tool is disabled: {tool_id}",
                error_code="tool_disabled",
                is_transient_error=False,
            )

        attempt = 0
        backoff = registration.initial_backoff_ms / 1000.0
        last_result: Optional[ToolResult] = None

        while attempt <= registration.max_retries:
            attempt += 1
            try:
                result = await asyncio.wait_for(
                    registration.adapter.execute(tool_input),
                    timeout=registration.timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                result = ToolResult(
                    tool_id=tool_id,
                    success=False,
                    error="Tool execution timeout.",
                    error_code="timeout",
                    is_transient_error=True,
                )
            last_result = result
            if result.success:
                return result
            if not result.is_transient_error or attempt > registration.max_retries:
                return result
            await asyncio.sleep(backoff)
            backoff *= 2

        return last_result or ToolResult(
            tool_id=tool_id,
            success=False,
            error="Unknown execution failure.",
            error_code="unknown",
            is_transient_error=False,
        )
