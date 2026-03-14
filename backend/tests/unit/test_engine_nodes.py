"""Tests for the refactored Reflex agent nodes.

All node methods are tested in isolation using fake LLMs and adapters.
The fake LLMs simulate bind_tools() by returning self from bind_tools()
and returning AIMessage-like objects from ainvoke().
"""
import asyncio
import json

import pytest

from contracts import ToolResult
from tools.adapters import McpToolAdapter
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fake LLMs
# ---------------------------------------------------------------------------

class _ToolCallLLM:
    """Fake LLM that returns a specific tool call when invoked."""

    def __init__(self, tool_name: str, tool_args: dict):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.bind_calls = 0

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        tool_name = self.tool_name
        tool_args = self.tool_args

        class _Resp:
            content = ""
            tool_calls = [{"name": tool_name, "args": tool_args, "id": "test-id", "type": "tool_call"}]

        return _Resp()


class _NoToolLLM:
    """Fake LLM that returns no tool calls (plain text answer)."""

    def __init__(self, content: str = "Direct answer."):
        self.content = content
        self.bind_calls = 0

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        content = self.content

        class _Resp:
            tool_calls = []

        _Resp.content = content
        return _Resp()


class _ErrorLLM:
    """Fake LLM that raises an exception during ainvoke."""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        raise RuntimeError("LLM unavailable")


class _CountingLLM:
    """Counts how many times ainvoke is called. Returns a plain answer."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1

        class _Resp:
            content = "Answer."
            tool_calls = []

        return _Resp()


# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------

class _FakeAdapter(McpToolAdapter):
    """Adapter that always succeeds. Optionally marks itself as not ready."""

    def __init__(self, tool_id: str, ready: bool = True):
        self.tool_id = tool_id
        self.ready = ready
        self.execute_calls = 0

    def readiness_error(self):
        return None if self.ready else f"{self.tool_id} adapter not ready"

    async def execute(self, tool_input):
        self.execute_calls += 1
        return ToolResult(tool_id=self.tool_id, success=True, output={"ok": True})

    def to_langchain_tool(self):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field as PydanticField

        class _Input(BaseModel):
            query: str = PydanticField(description=f"Query for {self.tool_id}")

        return StructuredTool(
            name=self.tool_id,
            description=f"Tool {self.tool_id}",
            args_schema=_Input,
            coroutine=self.execute,
        )


def _registry(tavily_ready: bool = True) -> ToolRegistry:
    r = ToolRegistry()
    r.register("tavily_search", _FakeAdapter("tavily_search", ready=tavily_ready), enabled=True, priority=10)
    return r


def _graph(llm=None, tavily_ready: bool = True):
    from orchestrator.engine import OrchestratorGraph
    return OrchestratorGraph(
        registry=_registry(tavily_ready=tavily_ready),
        llm=llm or _NoToolLLM(),
    )


# ---------------------------------------------------------------------------
# classify_node
# ---------------------------------------------------------------------------

def test_classify_node_returns_instant_reply_for_greeting():
    graph = _graph()
    state = {"user_message": "hello"}
    result = asyncio.run(graph.classify_node(state))
    assert result["instant_reply"] is not None
    assert "hello" in result["instant_reply"].lower()


def test_classify_node_returns_no_instant_reply_for_normal_query():
    graph = _graph()
    state = {"user_message": "What is the latest news?"}
    result = asyncio.run(graph.classify_node(state))
    assert result.get("instant_reply") is None


def test_classify_node_sets_final_response_for_greeting():
    graph = _graph()
    state = {"user_message": "merhaba"}
    result = asyncio.run(graph.classify_node(state))
    assert result.get("final_response") is not None
    assert result.get("validation_passed") is True


def test_classify_node_does_not_call_llm_for_greeting():
    llm = _CountingLLM()
    graph = _graph(llm=llm)
    asyncio.run(graph.classify_node({"user_message": "hi"}))
    assert llm.calls == 0


# ---------------------------------------------------------------------------
# tool_decision_node — LLM-native path
# ---------------------------------------------------------------------------

def test_tool_decision_selects_tavily_when_llm_returns_tavily_tool_call():
    llm = _ToolCallLLM(tool_name="tavily_search", tool_args={"query": "latest AI news"})
    graph = _graph(llm=llm)
    state = {
        "request_id": "r1",
        "user_message": "latest AI news",
        "history": [],
        "available_tools": ["tavily_search"],
    }
    result = asyncio.run(graph.tool_decision_node(state))
    assert result["selected_tool"] == "tavily_search"
    assert result["tool_args"] == {"query": "latest AI news"}


def test_tool_decision_stores_serialized_tool_call_message():
    llm = _ToolCallLLM(tool_name="tavily_search", tool_args={"query": "news"})
    graph = _graph(llm=llm)
    state = {
        "request_id": "r3",
        "user_message": "news",
        "history": [],
        "available_tools": ["tavily_search"],
    }
    result = asyncio.run(graph.tool_decision_node(state))
    assert result.get("tool_call_message") is not None
    parsed = json.loads(result["tool_call_message"])
    assert parsed[0]["name"] == "tavily_search"


def test_tool_decision_calls_bind_tools_with_available_tools():
    llm = _ToolCallLLM(tool_name="tavily_search", tool_args={"query": "test"})
    graph = _graph(llm=llm)
    state = {
        "request_id": "r4",
        "user_message": "test",
        "history": [],
        "available_tools": ["tavily_search"],
    }
    asyncio.run(graph.tool_decision_node(state))
    assert llm.bind_calls == 1


# ---------------------------------------------------------------------------
# tool_decision_node — keyword fallback path
# ---------------------------------------------------------------------------

def test_tool_decision_falls_back_to_keyword_for_tavily_when_llm_returns_no_calls():
    llm = _NoToolLLM()
    graph = _graph(llm=llm)
    state = {
        "request_id": "r6",
        "user_message": "Search the latest news today",
        "history": [],
        "available_tools": ["tavily_search"],
    }
    result = asyncio.run(graph.tool_decision_node(state))
    assert result["selected_tool"] == "tavily_search"


def test_tool_decision_returns_no_tool_when_no_match_and_no_llm_call():
    llm = _NoToolLLM()
    graph = _graph(llm=llm)
    state = {
        "request_id": "r7",
        "user_message": "What is 2 + 2?",
        "history": [],
        "available_tools": ["tavily_search"],
    }
    result = asyncio.run(graph.tool_decision_node(state))
    assert result["selected_tool"] is None


# ---------------------------------------------------------------------------
# tool_decision_node — readiness check
# ---------------------------------------------------------------------------

def test_tool_decision_clears_tool_when_tavily_not_ready():
    llm = _ToolCallLLM(tool_name="tavily_search", tool_args={"query": "news"})
    graph = _graph(llm=llm, tavily_ready=False)
    state = {
        "request_id": "r8",
        "user_message": "latest news today",
        "history": [],
        "available_tools": ["tavily_search"],
    }
    result = asyncio.run(graph.tool_decision_node(state))
    assert result["selected_tool"] is None


def test_tool_decision_handles_llm_error_gracefully():
    """If bind_tools() or ainvoke() raises, fall back to keyword routing."""
    graph = _graph(llm=_ErrorLLM())
    state = {
        "request_id": "r9",
        "user_message": "What is 2 + 2?",
        "history": [],
        "available_tools": ["tavily_search"],
    }
    result = asyncio.run(graph.tool_decision_node(state))
    assert result["selected_tool"] is None


# ---------------------------------------------------------------------------
# tool_execution_node
# ---------------------------------------------------------------------------

def test_tool_execution_uses_tool_args_query_when_present():
    graph = _graph()
    state = {
        "request_id": "r10",
        "session_id": "s1",
        "user_message": "original message",
        "selected_tool": "tavily_search",
        "tool_args": {"query": "specific query from llm"},
    }
    result = asyncio.run(graph.tool_execution_node(state))
    assert result["tool_result"]["success"] is True


def test_tool_execution_falls_back_to_user_message_when_no_tool_args():
    graph = _graph()
    state = {
        "request_id": "r11",
        "session_id": "s1",
        "user_message": "What is in the news?",
        "selected_tool": "tavily_search",
        "tool_args": None,
    }
    result = asyncio.run(graph.tool_execution_node(state))
    assert result["tool_result"]["success"] is True


def test_tool_execution_returns_none_when_no_tool_selected():
    graph = _graph()
    state = {
        "request_id": "r12",
        "session_id": "s1",
        "user_message": "hello",
        "selected_tool": None,
        "tool_args": None,
    }
    result = asyncio.run(graph.tool_execution_node(state))
    assert result["tool_result"] is None


# ---------------------------------------------------------------------------
# response_generation_node
# ---------------------------------------------------------------------------

def test_response_generation_returns_instant_reply_without_llm_call():
    llm = _CountingLLM()
    graph = _graph(llm=llm)
    state = {
        "request_id": "r13",
        "user_message": "hi",
        "history": [],
        "instant_reply": "Hi! How can I help you?",
        "tool_result": None,
    }
    result = asyncio.run(graph.response_generation_node(state))
    assert result == {}
    assert llm.calls == 0


def test_response_generation_produces_plain_text_not_json():
    llm = _NoToolLLM(content="This is a plain text answer.")
    graph = _graph(llm=llm)
    state = {
        "request_id": "r14",
        "user_message": "What is Python?",
        "history": [],
        "instant_reply": None,
        "tool_result": None,
    }
    result = asyncio.run(graph.response_generation_node(state))
    final = result["final_response"]
    assert final == "This is a plain text answer."
    assert not final.strip().startswith("{")


def test_response_generation_includes_tool_output_in_context():
    llm = _CountingLLM()
    graph = _graph(llm=llm)
    state = {
        "request_id": "r15",
        "user_message": "What is in the news?",
        "history": [],
        "instant_reply": None,
        "tool_result": {
            "tool_id": "tavily_search",
            "success": True,
            "output": {"query": "news", "content": ["Article 1", "Article 2"]},
        },
    }
    result = asyncio.run(graph.response_generation_node(state))
    assert result["final_response"] == "Answer."
    assert llm.calls == 1


def test_response_generation_shows_graceful_message_on_failed_tool():
    llm = _CountingLLM()
    graph = _graph(llm=llm)
    state = {
        "request_id": "r16",
        "user_message": "Search the web",
        "history": [],
        "instant_reply": None,
        "tool_result": {
            "tool_id": "tavily_search",
            "success": False,
            "error": "Tavily timeout",
            "error_code": "timeout",
        },
    }
    result = asyncio.run(graph.response_generation_node(state))
    assert llm.calls == 1
    assert result.get("final_response") is not None


def test_response_generation_handles_llm_exception_gracefully():
    graph = _graph(llm=_ErrorLLM())
    state = {
        "request_id": "r17",
        "user_message": "What is Python?",
        "history": [],
        "instant_reply": None,
        "tool_result": None,
    }
    result = asyncio.run(graph.response_generation_node(state))
    assert result.get("final_response") is not None
    assert len(result["final_response"]) > 0


# ---------------------------------------------------------------------------
# validation_node
# ---------------------------------------------------------------------------

def test_validation_passes_for_non_empty_response():
    graph = _graph()
    result = asyncio.run(graph.validation_node({"final_response": "Here is my answer."}))
    assert result["validation_passed"] is True


def test_validation_fails_for_empty_response():
    graph = _graph()
    result = asyncio.run(graph.validation_node({"final_response": ""}))
    assert result["validation_passed"] is False


def test_validation_fails_for_whitespace_only_response():
    graph = _graph()
    result = asyncio.run(graph.validation_node({"final_response": "   \n\t  "}))
    assert result["validation_passed"] is False


def test_validation_fails_for_none_response():
    graph = _graph()
    result = asyncio.run(graph.validation_node({"final_response": None}))
    assert result["validation_passed"] is False


# ---------------------------------------------------------------------------
# fallback_node
# ---------------------------------------------------------------------------

def test_fallback_node_sets_safe_final_response():
    graph = _graph()
    state = {"request_id": "r20", "retry_count": 0, "tool_result": None}
    result = asyncio.run(graph.fallback_node(state))
    assert result.get("final_response") is not None
    assert len(result["final_response"]) > 10


def test_fallback_node_marks_report_as_used():
    graph = _graph()
    state = {"request_id": "r21", "retry_count": 0, "tool_result": None}
    result = asyncio.run(graph.fallback_node(state))
    assert result["fallback_report"]["used"] is True
