from contracts import GraphState


def test_graphstate_has_new_fields():
    """New fields from refactor must be present in GraphState annotations."""
    annotations = GraphState.__annotations__
    assert "instant_reply" in annotations
    assert "tool_args" in annotations
    assert "tool_call_message" in annotations


def test_graphstate_removed_legacy_fields():
    """Legacy retry/routing fields must be gone."""
    annotations = GraphState.__annotations__
    assert "compressed_history" not in annotations
    assert "reasoning_notes" not in annotations
    assert "tool_decision" not in annotations
    assert "forced_tool_attempted" not in annotations
    assert "needs_forced_tool" not in annotations
    assert "needs_regeneration" not in annotations
    assert "max_retries" not in annotations
    assert "validation_report" not in annotations


def test_graphstate_keeps_core_fields():
    """Core fields that must remain."""
    annotations = GraphState.__annotations__
    for field in [
        "request_id", "session_id", "user_message", "history",
        "available_tools", "selected_tool", "tool_result",
        "final_response", "validation_passed", "fallback_report",
        "retry_count", "debug_metadata",
    ]:
        assert field in annotations, f"Missing core field: {field}"
