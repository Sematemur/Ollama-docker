import asyncio

from contracts import ToolResult
from tools.adapters import McpToolAdapter
from tools.registry import ToolRegistry


class _TransientAdapter(McpToolAdapter):
    def __init__(self):
        self.calls = 0

    async def execute(self, tool_input):
        self.calls += 1
        if self.calls < 2:
            return ToolResult(
                tool_id="dummy",
                success=False,
                error="temporary",
                error_code="temporary",
                is_transient_error=True,
            )
        return ToolResult(tool_id="dummy", success=True, output={"ok": True})


def test_registry_resolve_enabled_and_priority():
    registry = ToolRegistry()
    adapter = _TransientAdapter()
    registry.register("b_tool", adapter, enabled=True, priority=20)
    registry.register("a_tool", adapter, enabled=True, priority=10)
    registry.register("off_tool", adapter, enabled=False, priority=0)

    assert registry.resolve_available_tools() == ["a_tool", "b_tool"]


def test_registry_list_tools_returns_sorted_items():
    registry = ToolRegistry()
    adapter = _TransientAdapter()
    registry.register("b_tool", adapter, enabled=True, priority=20)
    registry.register("a_tool", adapter, enabled=True, priority=10)

    listed = registry.list_tools()
    assert [item.tool_id for item in listed] == ["a_tool", "b_tool"]


def test_registry_retries_transient_failures():
    registry = ToolRegistry()
    adapter = _TransientAdapter()
    registry.register(
        "dummy",
        adapter,
        enabled=True,
        priority=1,
        max_retries=2,
        initial_backoff_ms=1,
    )

    result = asyncio.run(registry.execute("dummy", {"query": "hello"}))
    assert result.success is True
    assert adapter.calls == 2


