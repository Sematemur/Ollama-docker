"""Single-agent Reflex orchestration graph built with LangGraph."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import settings
from contracts import FallbackReport, GraphState
from observability import (
    FALLBACK_EVENTS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    TOOL_CALLS_TOTAL,
    get_logger,
    hash_text,
    log_event,
)
from tools.registry import ToolRegistry

try:
    from opentelemetry import trace
except Exception:  # pragma: no cover
    trace = None

_HISTORY_TURNS = settings.orchestrator_history_turns
_TOOL_CONTEXT_MAX_CHARS = settings.orchestrator_tool_context_max_chars

_FALLBACK_MESSAGE = (
    "I could not generate a reliable response. "
    "Please clarify your question so I can respond accurately."
)
_ERROR_RESPONSE_MESSAGE = (
    "I encountered an error while generating a response. Please try again."
)
_NO_RESPONSE_MESSAGE = "I could not generate a response."

# ---------------------------------------------------------------------------
# Keyword markers (fallback routing when LLM returns no tool_calls)
# ---------------------------------------------------------------------------
WEB_SEARCH_STRONG_MARKERS = [
    "web", "internet", "online", "http", "https", "url",
    "source", "sources", "news", "headline", "headlines",
    "breaking", "guncel", "cevrimici",
]
WEB_SEARCH_ACTION_MARKERS = ["search", "look up", "lookup", "find", "ara", "bul"]
WEB_SEARCH_TOPIC_MARKERS = [
    "news", "current events", "latest news", "update", "updates",
    "price", "prices", "stock", "stocks", "weather", "score", "scores", "results",
]
WEB_SEARCH_TEMPORAL_MARKERS = ["latest", "current", "today", "now", "recent", "guncel", "son"]

GREETING_RESPONSES = {
    "hi": "Hi! How can I help you?",
    "hello": "Hello! How can I assist you today?",
    "hey": "Hey! How can I help you?",
    "selam": "Merhaba! Sana nasil yardimci olabilirim?",
    "merhaba": "Merhaba! Sana nasil yardimci olabilirim?",
    "good morning": "Good morning! How can I help you?",
    "good evening": "Good evening! How can I help you?",
    "what s up": "I am here and ready to help. What can I do for you?",
    "how are you": "I am doing well, thanks. How can I help you?",
}
GREETING_PREFIXES = {"hi", "hello", "hey", "selam", "merhaba"}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
TOOL_DECISION_SYSTEM_PROMPT = (
    "You are a routing assistant. Given a user message, decide whether to call a tool.\n\n"
    "Rules:\n"
    "- Use `tavily_search` for: current events, news, prices, live data, web search requests\n"
    "- Use neither for: greetings, arithmetic, general knowledge that does not require retrieval\n\n"
    "Reply only with a tool call or a brief direct answer. Do not explain your choice."
)

RESPONSE_GENERATION_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question clearly and concisely.\n"
    "If tool output is provided, use ONLY the facts present in it. Do NOT invent, extrapolate, or repeat data that is not explicitly provided.\n"
    "If the tool output does not contain enough information to fully answer the question, say so honestly instead of guessing.\n"
    "If the tool output indicates an error, acknowledge it gracefully without exposing technical details.\n"
    "Respond in plain text. Do not use JSON format."
)

# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _normalize_query_for_matching(query: str) -> str:
    raw = (query or "").lower()
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in raw)
    return " ".join(cleaned.split())


def _contains_query_marker(query: str, markers: List[str]) -> bool:
    normalized = f" {_normalize_query_for_matching(query)} "
    for marker in markers:
        nm = " ".join((marker or "").lower().split())
        if not nm:
            continue
        if " " in nm:
            if nm in normalized:
                return True
            continue
        if f" {nm} " in normalized:
            return True
    return False


def _is_web_search_query(query: str) -> bool:
    if _contains_query_marker(query, WEB_SEARCH_STRONG_MARKERS):
        return True
    has_action = _contains_query_marker(query, WEB_SEARCH_ACTION_MARKERS)
    has_topic = _contains_query_marker(query, WEB_SEARCH_TOPIC_MARKERS)
    has_temporal = _contains_query_marker(query, WEB_SEARCH_TEMPORAL_MARKERS)
    return has_topic and (has_action or has_temporal)


def _instant_greeting_response(query: str) -> Optional[str]:
    normalized = _normalize_query_for_matching(query)
    if not normalized:
        return None
    if normalized in GREETING_RESPONSES:
        return GREETING_RESPONSES[normalized]
    tokens = normalized.split()
    if len(tokens) <= 3 and tokens and tokens[0] in GREETING_PREFIXES:
        return GREETING_RESPONSES.get(tokens[0])
    return None


def _serialize_tool_context(tool_result: Optional[Dict[str, Any]]) -> str:
    if not tool_result:
        return ""
    raw = json.dumps(tool_result, ensure_ascii=True)
    if len(raw) <= _TOOL_CONTEXT_MAX_CHARS:
        return raw
    marker = "[TRUNCATED]"
    keep = max(_TOOL_CONTEXT_MAX_CHARS - len(marker), 0)
    return f"{raw[:keep]}{marker}"


class OrchestratorGraph:
    """Single-pass Reflex agent: classify → decide → execute → respond → validate."""

    def __init__(
        self,
        registry: ToolRegistry,
        llm: ChatOpenAI,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.logger = logger or get_logger()
        self.graph = self._build_graph()
        self.tracer = trace.get_tracer("chat.orchestrator") if trace else None

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("classify", self.classify_node)
        builder.add_node("tool_decision", self.tool_decision_node)
        builder.add_node("tool_execution", self.tool_execution_node)
        builder.add_node("response_generation", self.response_generation_node)
        builder.add_node("validation", self.validation_node)
        builder.add_node("fallback", self.fallback_node)

        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            lambda s: "end" if s.get("instant_reply") else "tool_decision",
            {"end": END, "tool_decision": "tool_decision"},
        )
        builder.add_conditional_edges(
            "tool_decision",
            lambda s: "tool_execution" if s.get("selected_tool") else "response_generation",
            {"tool_execution": "tool_execution", "response_generation": "response_generation"},
        )
        builder.add_edge("tool_execution", "response_generation")
        builder.add_edge("response_generation", "validation")
        builder.add_conditional_edges(
            "validation",
            lambda s: "end" if s.get("validation_passed") else "fallback",
            {"end": END, "fallback": "fallback"},
        )
        builder.add_edge("fallback", END)
        return builder.compile()

    async def run(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: str = "",
        request_id: Optional[str] = None,
    ) -> GraphState:
        state: GraphState = {
            "request_id": request_id or str(uuid.uuid4()),
            "session_id": session_id or "unknown",
            "user_message": user_message,
            "history": history or [],
            "available_tools": self.registry.resolve_available_tools(),
            "retry_count": 0,
            "validation_passed": False,
            "fallback_report": _model_dump(FallbackReport()),
            "debug_metadata": {},
        }
        start = time.perf_counter()
        if self.tracer:
            with self.tracer.start_as_current_span("chat.orchestrator.run"):
                result: GraphState = await self.graph.ainvoke(state)
        else:
            result = await self.graph.ainvoke(state)
        REQUEST_LATENCY_SECONDS.labels(route="/chat").observe(time.perf_counter() - start)
        return result

    async def classify_node(self, state: GraphState) -> Dict[str, Any]:
        """Instant greeting detection — no LLM call."""
        reply = _instant_greeting_response(state["user_message"])
        if reply:
            return {"instant_reply": reply, "final_response": reply, "validation_passed": True}
        return {"instant_reply": None}

    async def tool_decision_node(self, state: GraphState) -> Dict[str, Any]:
        """LLM decides which tool to call via bind_tools(). Keyword fallback if no tool_calls."""
        available = state.get("available_tools", [])
        selected_tool: Optional[str] = None
        tool_args: Optional[Dict[str, Any]] = None
        tool_call_message: Optional[str] = None

        lc_tools = []
        for tid in available:
            reg = self.registry.get(tid)
            if reg and reg.enabled:
                lc_tool_fn = getattr(reg.adapter, "to_langchain_tool", None)
                if callable(lc_tool_fn):
                    lc_tools.append(lc_tool_fn())

        if lc_tools:
            try:
                llm_with_tools = self.llm.bind_tools(lc_tools)
                history = state.get("history", [])[-_HISTORY_TURNS:]
                user_prompt = state["user_message"]
                if history:
                    user_prompt = (
                        f"Recent conversation: {json.dumps(history, ensure_ascii=True)}\n\n"
                        f"User: {state['user_message']}"
                    )
                ai_response = await llm_with_tools.ainvoke([
                    SystemMessage(content=TOOL_DECISION_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ])
                calls = getattr(ai_response, "tool_calls", None) or []
                if calls:
                    first = calls[0]
                    name = first.get("name") if isinstance(first, dict) else getattr(first, "name", None)
                    args = first.get("args") if isinstance(first, dict) else getattr(first, "args", {})
                    if name in available:
                        selected_tool = name
                        tool_args = dict(args) if args else {}
                        tool_call_message = json.dumps(calls, default=str)
                        log_event(self.logger, logging.INFO, "tool_selected_llm", {
                            "request_id": state["request_id"],
                            "tool_id": selected_tool,
                        })
            except Exception as exc:
                log_event(self.logger, logging.WARNING, "bind_tools_failed", {
                    "request_id": state["request_id"],
                    "error": str(exc),
                })

        if selected_tool is None:
            query = state["user_message"]
            if _is_web_search_query(query) and "tavily_search" in available:
                selected_tool = "tavily_search"
                tool_args = {"query": query}
                log_event(self.logger, logging.INFO, "tool_selected_keyword", {
                    "request_id": state["request_id"],
                    "tool_id": selected_tool,
                    "reason": "web_search_keyword_match",
                })

        if selected_tool:
            reg = self.registry.get(selected_tool)
            checker = getattr(getattr(reg, "adapter", None), "readiness_error", None)
            if callable(checker):
                err = checker()
                if err:
                    log_event(self.logger, logging.WARNING, "tool_not_ready", {
                        "request_id": state["request_id"],
                        "tool_id": selected_tool,
                        "error": err,
                    })
                    selected_tool = None
                    tool_args = None

        return {
            "selected_tool": selected_tool,
            "tool_args": tool_args,
            "tool_call_message": tool_call_message,
        }

    async def tool_execution_node(self, state: GraphState) -> Dict[str, Any]:
        """Execute the selected tool through the registry (retry + timeout preserved)."""
        tool_id = state.get("selected_tool")
        if not tool_id:
            return {"tool_result": None}

        tool_args = state.get("tool_args") or {}
        query = tool_args.get("query") or state["user_message"]
        tool_input = {"query": query, "session_id": state["session_id"]}

        if self.tracer:
            with self.tracer.start_as_current_span(f"tool.{tool_id}.execute"):
                result = await self.registry.execute(tool_id, tool_input)
        else:
            result = await self.registry.execute(tool_id, tool_input)

        TOOL_CALLS_TOTAL.labels(
            tool_id=tool_id, status="success" if result.success else "failure"
        ).inc()
        log_event(self.logger, logging.INFO, "tool_call", {
            "request_id": state["request_id"],
            "tool_id": tool_id,
            "success": result.success,
            "latency_ms": result.latency_ms,
            "error_code": result.error_code,
            "tool_args_hash": hash_text(query),
        })
        return {"tool_result": _model_dump(result)}

    async def response_generation_node(self, state: GraphState) -> Dict[str, Any]:
        """Generate plain-text answer. Uses tool output if available."""
        if state.get("instant_reply"):
            return {}

        tool_result = state.get("tool_result")
        history = state.get("history", [])

        user_prompt = f"User: {state['user_message']}\n"
        if history:
            user_prompt += f"Recent conversation: {json.dumps(history[-_HISTORY_TURNS:], ensure_ascii=True)}\n"

        if tool_result:
            if not tool_result.get("success"):
                user_prompt += "Note: I'm currently unable to retrieve external information. Please try again later.\n"
            else:
                ctx = _serialize_tool_context(tool_result)
                if ctx:
                    user_prompt += f"Tool output: {ctx}\n"

        try:
            llm_response = await self.llm.ainvoke([
                SystemMessage(content=RESPONSE_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            final_response = (llm_response.content or "").strip()
        except Exception as exc:
            log_event(self.logger, logging.ERROR, "response_generation_failed", {
                "request_id": state["request_id"],
                "error": str(exc),
            })
            final_response = _ERROR_RESPONSE_MESSAGE

        return {"final_response": final_response, "raw_model_response": final_response}

    async def validation_node(self, state: GraphState) -> Dict[str, Any]:
        """Simple non-empty check."""
        final_response = state.get("final_response") or ""
        return {"validation_passed": bool(final_response.strip())}

    async def fallback_node(self, state: GraphState) -> Dict[str, Any]:
        """Last resort: safe message + metrics."""
        FALLBACK_EVENTS_TOTAL.inc()
        log_event(self.logger, logging.ERROR, "fallback_used", {
            "request_id": state["request_id"],
            "retry_count": state.get("retry_count", 0),
        })
        return {
            "fallback_report": _model_dump(FallbackReport(
                used=True, reason="Validation failed: empty response.", strategy="clarification"
            )),
            "final_response": _FALLBACK_MESSAGE,
        }
