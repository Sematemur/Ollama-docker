"""Shared typed contracts used by orchestration, tools, and API layers."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ToolDecision(BaseModel):
    """Tool selection decision produced by orchestration."""

    tool_id: Optional[str] = None
    reason: str = ""
    must_use_tool: bool = False
    confidence: float = 0.0


class ToolResult(BaseModel):
    """Standard result envelope returned by tool adapters."""

    tool_id: str
    success: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    error_code: Optional[str] = None
    is_transient_error: bool = False
    latency_ms: Optional[int] = None


class ValidationReport(BaseModel):
    """Validation outcome for response/tool behavior."""

    passed: bool
    errors: List[str] = Field(default_factory=list)
    should_force_tool: bool = False
    should_regenerate_fragment: bool = False


class FallbackReport(BaseModel):
    """Fallback metadata when normal generation fails."""

    used: bool = False
    reason: Optional[str] = None
    strategy: Optional[
        Literal["safe_default", "clarification", "cached_response"]
    ] = None


class ResponseEnvelope(BaseModel):
    """Schema for the model's structured response payload."""

    answer: str
    needs_clarification: bool = False
    confidence: float = 0.5
    citations: List[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    """LangGraph state container — single-pass Reflex agent."""

    # Identity
    request_id: str
    session_id: str

    # Input
    user_message: str
    history: List[Dict[str, str]]

    # Fast path (classify_node)
    instant_reply: Optional[str]

    # Tool decision (tool_decision_node)
    available_tools: List[str]
    selected_tool: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    tool_call_message: Optional[str]       # json.dumps(ai_response.tool_calls)

    # Tool execution (tool_execution_node)
    tool_result: Optional[Dict[str, Any]]  # _model_dump(ToolResult)

    # Response (response_generation_node)
    raw_model_response: Optional[str]
    final_response: Optional[str]

    # Validation
    validation_passed: bool

    # Fallback
    fallback_report: Dict[str, Any]        # _model_dump(FallbackReport)

    # Observability
    retry_count: int
    debug_metadata: Dict[str, Any]
