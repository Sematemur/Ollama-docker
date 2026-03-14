# Reflex Agent — LangGraph Native Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace brittle keyword-based tool selection with LLM-native `bind_tools()` routing, fix GraphRAG configuration, and simplify the orchestrator to a clean single-pass Reflex agent.

**Architecture:** `classify_node` handles greetings instantly → `tool_decision_node` calls `llm.bind_tools()` (with keyword fallback) → `tool_execution_node` runs the tool via registry → `response_generation_node` produces plain-text answer → `validation_node` checks non-empty → done. No retry loops.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, LangChain (langchain-core, langchain-openai), Ollama (llama3.1 + nomic-embed-text), LiteLLM proxy, Docker Compose, pytest.

**Spec:** `docs/superpowers/specs/2026-03-13-reflex-agent-langgraph-tool-calling-design.md`

---

## Chunk 1: Config Fixes + GraphState Simplification

### Task 1: Fix configuration files

**Files:**
- Modify: `.env`
- Modify: `.env.example`
- Modify: `litellm-config.yaml`

No unit tests for config — verified manually via `/tools/status` endpoint.

- [ ] **Step 1: Fix `.env` — open the file and apply these changes**

  ```
  LITELLM_MAX_TOKENS=2048          # was 320
  GRAPHRAG_EMBEDDING_MODEL=ollama/nomic-embed-text   # was text-embedding-3-small
  GRAPHRAG_CHAT_MODEL=ollama/llama3.1:latest         # was gpt-4.1-mini
  GRAPHRAG_API_KEY=<your-litellm-master-key>         # must be litellm-config.yaml master_key value
  GRAPHRAG_QUERY_TIMEOUT_SECONDS=30                  # was 25
  GRAPHRAG_TIMEOUT_MS=35000                          # NEW variable — add it
  ```

  Leave `GRAPHRAG_API_BASE=http://litellm:4000` unchanged (already correct).

- [ ] **Step 2: Fix `.env.example` — apply the same changes as `.env` but use placeholder for the key**

  ```
  LITELLM_MAX_TOKENS=2048
  GRAPHRAG_EMBEDDING_MODEL=ollama/nomic-embed-text
  GRAPHRAG_CHAT_MODEL=ollama/llama3.1:latest
  GRAPHRAG_API_KEY=sk-<your-litellm-master-key>
  GRAPHRAG_QUERY_TIMEOUT_SECONDS=30
  GRAPHRAG_TIMEOUT_MS=35000                          # NEW — add this line
  LITELLM_MASTER_KEY=sk-<your-litellm-master-key>   # NEW — add this line
  ```

- [ ] **Step 3: Fix `litellm-config.yaml` — replace hardcoded master key with env var**

  Change:
  ```yaml
  general_settings:
    master_key: REDACTED
  ```
  To:
  ```yaml
  general_settings:
    master_key: ${LITELLM_MASTER_KEY}
  ```

  Add `LITELLM_MASTER_KEY=<actual-key-value>` to your `.env` (the same value that was hardcoded).

- [ ] **Step 4: Verify config is read correctly**

  ```bash
  docker compose up litellm postgres redis -d
  # wait ~20 seconds for litellm to be healthy
  curl http://localhost:8023/health/liveliness
  # Expected: {"status": "healthy"} or similar
  ```

---

### Task 2: Simplify GraphState in contracts.py

**Files:**
- Modify: `backend/contracts.py`

- [ ] **Step 1: Write the failing test for new GraphState shape**

  Create `backend/tests/unit/test_contracts.py`:

  ```python
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
  ```

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  cd backend && python -m pytest tests/unit/test_contracts.py -v
  # Expected: FAIL — legacy fields still present
  ```

- [ ] **Step 3: Replace GraphState in `backend/contracts.py`**

  Replace the existing `GraphState` class (lines 59-83) with:

  ```python
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
  ```

  Keep `ToolDecision`, `ToolResult`, `ValidationReport`, `FallbackReport`, `ResponseEnvelope` unchanged (they stay in contracts.py; ResponseEnvelope is no longer used in the execution path but kept for backwards compatibility).

- [ ] **Step 4: Run test to verify it passes**

  ```bash
  cd backend && python -m pytest tests/unit/test_contracts.py -v
  # Expected: PASS (3 tests)
  ```

---

## Chunk 2: Tool LangChain Schema Methods

### Task 3: Add `to_langchain_tool()` to adapters

**Files:**
- Modify: `backend/tools/adapters.py`
- Create: `backend/tests/unit/test_adapter_schemas.py`

- [ ] **Step 1: Write failing tests for `to_langchain_tool()`**

  Create `backend/tests/unit/test_adapter_schemas.py`:

  ```python
  import pytest
  from tools.adapters import TavilyDirectAdapter, GraphRAGAdapter


  def test_tavily_adapter_has_to_langchain_tool():
      adapter = TavilyDirectAdapter(api_key="test-key")
      assert hasattr(adapter, "to_langchain_tool")
      assert callable(adapter.to_langchain_tool)


  def test_tavily_langchain_tool_has_correct_name():
      adapter = TavilyDirectAdapter(api_key="test-key")
      tool = adapter.to_langchain_tool()
      assert tool.name == "tavily_search"


  def test_tavily_langchain_tool_has_query_in_schema():
      adapter = TavilyDirectAdapter(api_key="test-key")
      tool = adapter.to_langchain_tool()
      schema = tool.args_schema.model_json_schema()
      assert "query" in schema["properties"]


  def test_tavily_langchain_tool_description_mentions_web_search():
      adapter = TavilyDirectAdapter(api_key="test-key")
      tool = adapter.to_langchain_tool()
      assert "web" in tool.description.lower() or "search" in tool.description.lower()


  def test_graphrag_adapter_has_to_langchain_tool():
      adapter = GraphRAGAdapter(workspace_root="./data/graphrag")
      assert hasattr(adapter, "to_langchain_tool")
      assert callable(adapter.to_langchain_tool)


  def test_graphrag_langchain_tool_has_correct_name():
      adapter = GraphRAGAdapter(workspace_root="./data/graphrag")
      tool = adapter.to_langchain_tool()
      assert tool.name == "graphrag"


  def test_graphrag_langchain_tool_has_query_in_schema():
      adapter = GraphRAGAdapter(workspace_root="./data/graphrag")
      tool = adapter.to_langchain_tool()
      schema = tool.args_schema.model_json_schema()
      assert "query" in schema["properties"]


  def test_graphrag_langchain_tool_description_mentions_document():
      adapter = GraphRAGAdapter(workspace_root="./data/graphrag")
      tool = adapter.to_langchain_tool()
      assert "document" in tool.description.lower() or "knowledge" in tool.description.lower()


  def test_both_tools_return_different_names():
      tavily = TavilyDirectAdapter(api_key="test-key").to_langchain_tool()
      graphrag = GraphRAGAdapter(workspace_root="./data/graphrag").to_langchain_tool()
      assert tavily.name != graphrag.name
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  cd backend && python -m pytest tests/unit/test_adapter_schemas.py -v
  # Expected: FAIL — to_langchain_tool() does not exist yet
  ```

- [ ] **Step 3: Add imports at the top of `backend/tools/adapters.py`**

  Add after the existing imports (around line 14):

  ```python
  from langchain_core.tools import StructuredTool
  from pydantic import Field
  ```

  Note: `BaseModel` is already imported via `from contracts import ToolResult` (pydantic is available).
  Add explicitly: `from pydantic import BaseModel, Field`

- [ ] **Step 4: Add `to_langchain_tool()` to `TavilyDirectAdapter`**

  Add this method inside `TavilyDirectAdapter`, after the `readiness_error` method (around line 64):

  ```python
  def to_langchain_tool(self) -> StructuredTool:
      """Return a LangChain StructuredTool for use with llm.bind_tools().

      The schema is passed to the LLM to describe when/how to call this tool.
      Actual execution goes through ToolRegistry.execute(), not this StructuredTool.
      """
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
  ```

- [ ] **Step 5: Add `to_langchain_tool()` to `GraphRAGAdapter`**

  Add this method inside `GraphRAGAdapter`, after the `readiness_error` method (around line 391):

  ```python
  def to_langchain_tool(self) -> StructuredTool:
      """Return a LangChain StructuredTool for use with llm.bind_tools().

      The schema is passed to the LLM to describe when/how to call this tool.
      Actual execution goes through ToolRegistry.execute(), not this StructuredTool.
      """
      class GraphRAGInput(BaseModel):
          query: str = Field(description="Question about the uploaded thesis, project documents, or knowledge base")

      return StructuredTool(
          name="graphrag",
          description=(
              "Query the uploaded document knowledge base (PDFs, thesis). "
              "Use when the user asks about the project, thesis content, medical imaging, or uploaded documents."
          ),
          args_schema=GraphRAGInput,
          coroutine=self.execute,
      )
  ```

- [ ] **Step 6: Run tests to verify they pass**

  ```bash
  cd backend && python -m pytest tests/unit/test_adapter_schemas.py -v
  # Expected: PASS (9 tests)
  ```

- [ ] **Step 7: Verify existing adapter tests still pass**

  ```bash
  cd backend && python -m pytest tests/unit/test_tavily_adapter.py tests/unit/test_graphrag_adapter.py tests/unit/test_tool_registry.py -v
  # Expected: all PASS — to_langchain_tool() is additive, execute() is unchanged
  ```

---

## Chunk 3: Engine Rewrite

### Task 4: Write failing tests for the new engine nodes

**Files:**
- Create: `backend/tests/unit/test_engine_nodes.py`

- [ ] **Step 1: Create test file with fake LLM helpers**

  Create `backend/tests/unit/test_engine_nodes.py`:

  ```python
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


  def _registry(graphrag_ready: bool = True) -> ToolRegistry:
      r = ToolRegistry()
      r.register("tavily_search", _FakeAdapter("tavily_search"), enabled=True, priority=10)
      r.register("graphrag", _FakeAdapter("graphrag", ready=graphrag_ready), enabled=True, priority=20)
      return r


  def _graph(llm=None, graphrag_ready: bool = True):
      from orchestrator.engine import OrchestratorGraph
      return OrchestratorGraph(
          registry=_registry(graphrag_ready=graphrag_ready),
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
      state = {"user_message": "What is the thesis about?"}
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
          "available_tools": ["tavily_search", "graphrag"],
      }
      result = asyncio.run(graph.tool_decision_node(state))
      assert result["selected_tool"] == "tavily_search"
      assert result["tool_args"] == {"query": "latest AI news"}


  def test_tool_decision_selects_graphrag_when_llm_returns_graphrag_tool_call():
      llm = _ToolCallLLM(tool_name="graphrag", tool_args={"query": "thesis methodology"})
      graph = _graph(llm=llm)
      state = {
          "request_id": "r2",
          "user_message": "What does the thesis say about methodology?",
          "history": [],
          "available_tools": ["tavily_search", "graphrag"],
      }
      result = asyncio.run(graph.tool_decision_node(state))
      assert result["selected_tool"] == "graphrag"


  def test_tool_decision_stores_serialized_tool_call_message():
      llm = _ToolCallLLM(tool_name="tavily_search", tool_args={"query": "news"})
      graph = _graph(llm=llm)
      state = {
          "request_id": "r3",
          "user_message": "news",
          "history": [],
          "available_tools": ["tavily_search", "graphrag"],
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
          "available_tools": ["tavily_search", "graphrag"],
      }
      asyncio.run(graph.tool_decision_node(state))
      assert llm.bind_calls == 1


  # ---------------------------------------------------------------------------
  # tool_decision_node — keyword fallback path
  # ---------------------------------------------------------------------------

  def test_tool_decision_falls_back_to_keyword_for_graphrag_when_llm_returns_no_calls():
      llm = _NoToolLLM()
      graph = _graph(llm=llm)
      state = {
          "request_id": "r5",
          "user_message": "What does the thesis say about medical imaging?",
          "history": [],
          "available_tools": ["tavily_search", "graphrag"],
      }
      result = asyncio.run(graph.tool_decision_node(state))
      # "thesis" matches PROJECT_RAG_MARKERS → graphrag selected via keyword fallback
      assert result["selected_tool"] == "graphrag"


  def test_tool_decision_falls_back_to_keyword_for_tavily_when_llm_returns_no_calls():
      llm = _NoToolLLM()
      graph = _graph(llm=llm)
      state = {
          "request_id": "r6",
          "user_message": "Search the latest news today",
          "history": [],
          "available_tools": ["tavily_search", "graphrag"],
      }
      result = asyncio.run(graph.tool_decision_node(state))
      # "search" (action) + "news" (topic) → tavily selected via keyword fallback
      assert result["selected_tool"] == "tavily_search"


  def test_tool_decision_returns_no_tool_when_no_match_and_no_llm_call():
      llm = _NoToolLLM()
      graph = _graph(llm=llm)
      state = {
          "request_id": "r7",
          "user_message": "What is 2 + 2?",
          "history": [],
          "available_tools": ["tavily_search", "graphrag"],
      }
      result = asyncio.run(graph.tool_decision_node(state))
      assert result["selected_tool"] is None


  # ---------------------------------------------------------------------------
  # tool_decision_node — readiness check
  # ---------------------------------------------------------------------------

  def test_tool_decision_clears_tool_when_graphrag_not_ready():
      llm = _ToolCallLLM(tool_name="graphrag", tool_args={"query": "thesis"})
      graph = _graph(llm=llm, graphrag_ready=False)
      state = {
          "request_id": "r8",
          "user_message": "What is the thesis about?",
          "history": [],
          "available_tools": ["tavily_search", "graphrag"],
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
          "available_tools": ["tavily_search", "graphrag"],
      }
      # Should not raise — graceful fallback to keyword (no match → None)
      result = asyncio.run(graph.tool_decision_node(state))
      assert result["selected_tool"] is None


  # ---------------------------------------------------------------------------
  # tool_execution_node
  # ---------------------------------------------------------------------------

  def test_tool_execution_uses_tool_args_query_when_present():
      graph = _graph()
      adapter = _registry().get("tavily_search").adapter
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
          "user_message": "What is in the thesis?",
          "selected_tool": "graphrag",
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
      assert result == {}   # instant_reply already set; node returns empty update
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
      # Must be plain text, not JSON
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
          "user_message": "What does the PDF say?",
          "history": [],
          "instant_reply": None,
          "tool_result": {
              "tool_id": "graphrag",
              "success": False,
              "error": "GraphRAG index missing",
              "error_code": "index_missing",
          },
      }
      result = asyncio.run(graph.response_generation_node(state))
      # LLM still called (to generate answer incorporating the graceful error note)
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
      # Must not raise — returns an error message string
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
  ```

- [ ] **Step 2: Run tests to verify they all fail (old engine does not have these methods)**

  ```bash
  cd backend && python -m pytest tests/unit/test_engine_nodes.py -v 2>&1 | head -60
  # Expected: FAIL — classify_node, tool_decision_node not found or wrong behavior
  ```

---

### Task 5: Rewrite `backend/orchestrator/engine.py`

**Files:**
- Modify: `backend/orchestrator/engine.py`

This is the main implementation task. Replace the entire file with the new implementation.

- [ ] **Step 1: Replace the imports section at the top of `engine.py`**

  Replace lines 1–41 with:

  ```python
  """Single-agent Reflex orchestration graph built with LangGraph."""

  from __future__ import annotations

  import json
  import logging
  import os
  import time
  import uuid
  from typing import Any, Dict, List, Optional

  from langchain_core.messages import HumanMessage, SystemMessage
  from langchain_openai import ChatOpenAI
  from langgraph.graph import END, START, StateGraph

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
  ```

- [ ] **Step 2: Replace the constants and helper section**

  Replace lines 43–250 (all the markers, prompts, and helper functions) with:

  ```python
  # ---------------------------------------------------------------------------
  # Keyword markers (kept as fallback routing when LLM returns no tool_calls)
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
  PROJECT_RAG_MARKERS = [
      "bitirme projesi", "graduation project", "final project", "capstone",
      "thesis", "tez", "project report", "medical imaging", "automatic reporting",
      "medikal goruntulemede", "yapay zeka destekli otomatik raporlama",
      "bilgi tabani", "knowledge base", "knowledge-base", "graphrag", "rag",
      "uploaded pdf", "uploaded document", "uploaded file", "uploaded thesis",
      "my thesis", "my pdf", "this thesis", "this pdf", "the thesis", "the pdf",
      "this document", "the document", "my document", "document i uploaded",
      "pdf i uploaded", "dokuman",
  ]

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
      "- Use `graphrag` for: questions about the uploaded thesis, project documents, or knowledge base content\n"
      "- Use neither for: greetings, arithmetic, general knowledge that does not require retrieval\n\n"
      "Reply only with a tool call or a brief direct answer. Do not explain your choice."
  )

  RESPONSE_GENERATION_SYSTEM_PROMPT = (
      "You are a helpful AI assistant. Answer the user's question clearly and concisely.\n"
      "If tool output is provided, use it to inform your answer.\n"
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


  def _is_project_rag_query(query: str) -> bool:
      return _contains_query_marker(query, PROJECT_RAG_MARKERS)


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


  def _graceful_tool_failure_message(tool_id: Optional[str]) -> str:
      if (tool_id or "").strip().lower() == "graphrag":
          return (
              "I'm currently unable to retrieve information from the knowledge base. "
              "Please try again later."
          )
      return (
          "I'm currently unable to retrieve external information right now. "
          "Please try again later."
      )


  def _serialize_tool_context(tool_result: Optional[Dict[str, Any]], limit: int = 2500) -> str:
      if not tool_result:
          return ""
      raw = json.dumps(tool_result, ensure_ascii=True)
      if len(raw) <= limit:
          return raw
      marker = "[TRUNCATED]"
      keep = max(limit - len(marker), 0)
      return f"{raw[:keep]}{marker}"
  ```

- [ ] **Step 3: Replace the `OrchestratorGraph` class**

  Replace everything from `class OrchestratorGraph:` to the end of the file with:

  ```python
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

      # ------------------------------------------------------------------
      # Nodes
      # ------------------------------------------------------------------

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

          # Build LangChain schemas from adapters for bind_tools()
          lc_tools = []
          for tid in available:
              reg = self.registry.get(tid)
              if reg and reg.enabled:
                  lc_tool_fn = getattr(reg.adapter, "to_langchain_tool", None)
                  if callable(lc_tool_fn):
                      lc_tools.append(lc_tool_fn())

          # Primary path: LLM-native tool calling
          if lc_tools:
              try:
                  llm_with_tools = self.llm.bind_tools(lc_tools)
                  history_turns = int(os.getenv("ORCHESTRATOR_HISTORY_TURNS", "6"))
                  history = state.get("history", [])[-history_turns:]
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

          # Fallback path: keyword routing
          if selected_tool is None:
              query = state["user_message"]
              if _is_project_rag_query(query) and "graphrag" in available:
                  selected_tool = "graphrag"
                  tool_args = {"query": query}
                  log_event(self.logger, logging.INFO, "tool_selected_keyword", {
                      "request_id": state["request_id"],
                      "tool_id": selected_tool,
                      "reason": "project_rag_keyword_match",
                  })
              elif _is_web_search_query(query) and "tavily_search" in available:
                  selected_tool = "tavily_search"
                  tool_args = {"query": query}
                  log_event(self.logger, logging.INFO, "tool_selected_keyword", {
                      "request_id": state["request_id"],
                      "tool_id": selected_tool,
                      "reason": "web_search_keyword_match",
                  })

          # Readiness check: clear tool if adapter reports not ready
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
          # Greeting already handled in classify_node — nothing to do here
          if state.get("instant_reply"):
              return {}

          tool_result = state.get("tool_result")
          history = state.get("history", [])
          history_turns = int(os.getenv("ORCHESTRATOR_HISTORY_TURNS", "6"))

          user_prompt = f"User: {state['user_message']}\n"
          if history:
              user_prompt += f"Recent conversation: {json.dumps(history[-history_turns:], ensure_ascii=True)}\n"

          if tool_result:
              if not tool_result.get("success"):
                  tool_id = str(tool_result.get("tool_id") or "").strip().lower() or None
                  user_prompt += f"Note: {_graceful_tool_failure_message(tool_id)}\n"
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
              final_response = "I encountered an error while generating a response. Please try again."

          return {"final_response": final_response, "raw_model_response": final_response}

      async def validation_node(self, state: GraphState) -> Dict[str, Any]:
          """Simple non-empty check. No retry loops."""
          final_response = state.get("final_response") or ""
          return {"validation_passed": bool(final_response.strip())}

      async def fallback_node(self, state: GraphState) -> Dict[str, Any]:
          """Last resort: safe message + metrics."""
          FALLBACK_EVENTS_TOTAL.inc()
          safe = (
              "I could not generate a reliable response. "
              "Please clarify your question so I can respond accurately."
          )
          log_event(self.logger, logging.ERROR, "fallback_used", {
              "request_id": state["request_id"],
              "retry_count": state.get("retry_count", 0),
          })
          return {
              "fallback_report": _model_dump(FallbackReport(
                  used=True, reason="Validation failed: empty response.", strategy="clarification"
              )),
              "final_response": safe,
          }
  ```

- [ ] **Step 4: Run the new engine node tests**

  ```bash
  cd backend && python -m pytest tests/unit/test_engine_nodes.py -v
  # Expected: all PASS
  ```

- [ ] **Step 5: Run the adapter schema tests (should still pass)**

  ```bash
  cd backend && python -m pytest tests/unit/test_adapter_schemas.py tests/unit/test_tool_registry.py -v
  # Expected: all PASS
  ```

---

### Task 6: Replace the old engine validation tests

The old `test_validation.py` tests the removed nodes (`tool_selection_node`, retry loops, JSON envelope validation). Delete it and keep only the new tests.

**Files:**
- Delete: `backend/tests/unit/test_validation.py`

- [ ] **Step 1: Delete the old test file**

  ```bash
  rm backend/tests/unit/test_validation.py
  ```

- [ ] **Step 2: Run full unit test suite to confirm no regressions**

  ```bash
  cd backend && python -m pytest tests/unit/ -v
  # Expected: all PASS
  # Files covered: test_contracts.py, test_adapter_schemas.py, test_engine_nodes.py,
  #                test_tavily_adapter.py, test_graphrag_adapter.py, test_tool_registry.py
  ```

---

## Chunk 4: agent.py Fix + End-to-End Verification

### Task 7: Fix debug_data in `backend/agent.py`

**Files:**
- Modify: `backend/agent.py`

The `debug_data` block reads `state.get("tool_decision")` which no longer exists. Fix it to read `selected_tool` directly.

- [ ] **Step 1: Locate the debug_data block in `agent.py` (around lines 141–149)**

  Current code:
  ```python
  if debug:
      decision = state.get("tool_decision") or {}
      validation = state.get("validation_report") or {}
      debug_data = {
          "selected_tool": decision.get("tool_id"),
          "retries": state.get("retry_count", 0),
          "fallback_used": fallback_report.get("used", False),
          "validation_passed": validation.get("passed", False),
          "cache_hit": False,
      }
  ```

- [ ] **Step 2: Replace with updated version**

  ```python
  if debug:
      debug_data = {
          "selected_tool": state.get("selected_tool"),
          "retries": state.get("retry_count", 0),
          "fallback_used": fallback_report.get("used", False),
          "validation_passed": state.get("validation_passed", False),
          "cache_hit": False,
      }
  ```

- [ ] **Step 3: Run full test suite one final time**

  ```bash
  cd backend && python -m pytest tests/unit/ -v
  # Expected: all PASS
  ```

---

### Task 8: End-to-End Verification

No code changes — manual verification that the full system works.

- [ ] **Step 1: Start the stack**

  ```bash
  docker compose up --build -d
  # Wait for all services to be healthy (~60s)
  docker compose ps
  # All services should show "healthy" or "running"
  ```

- [ ] **Step 2: Verify tool readiness**

  ```bash
  curl http://localhost:8000/tools/status
  # Expected: tavily_search ready:true, graphrag may show needs_index error (that's OK)
  ```

- [ ] **Step 3: Build the GraphRAG index (one-time, ~5-15 minutes)**

  ```bash
  # First verify GRAPHRAG_BOOTSTRAP_PDF_PATH is set in .env and the PDF exists
  curl -X POST http://localhost:8000/rag/index
  # Expected: {"workspace": "...", "stdout": "...", "stderr": "..."}
  # Wait for completion — this runs graphrag index which takes several minutes
  ```

- [ ] **Step 4: Verify GraphRAG is ready**

  ```bash
  curl http://localhost:8000/rag/status
  # Expected: needs_index:false, ready:true
  ```

- [ ] **Step 5: Test Tavily tool selection**

  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "What is in the tech news today?", "debug": true}'
  # Expected: debug_metadata.selected_tool == "tavily_search"
  # Response should contain current web content
  ```

- [ ] **Step 6: Test GraphRAG tool selection**

  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Which LLM models were used in the graduation project?", "debug": true}'
  # Expected: debug_metadata.selected_tool == "graphrag"
  # Response should reference content from the thesis PDF
  ```

- [ ] **Step 7: Test direct answer (no tool)**

  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "What is 5 times 7?"}'
  # Expected: response is "35" or similar, no tool latency
  ```

- [ ] **Step 8: Test greeting fast path**

  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "hi"}'
  # Expected: instant response, "Hi! How can I help you?" or similar
  ```

- [ ] **Step 9: Test keyword fallback (Tavily)**

  For the US-Iran war query that previously failed:

  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "What day of the war between the United States and Iran are we currently on?", "debug": true}'
  # Expected: debug_metadata.selected_tool == "tavily_search" (LLM should call it)
  # OR if LLM doesn't call it: selected_tool == null (LLM answers from knowledge)
  # Either way: NOT a JSON error, NOT a truncation failure
  ```

---

## Summary: Files Changed

| File | Change |
|------|--------|
| `.env` | 5 config fixes |
| `.env.example` | Same 5 fixes + LITELLM_MASTER_KEY placeholder |
| `litellm-config.yaml` | master_key → ${LITELLM_MASTER_KEY} |
| `backend/contracts.py` | Replace GraphState (8 fields removed, 3 added) |
| `backend/tools/adapters.py` | Add to_langchain_tool() to both adapters |
| `backend/orchestrator/engine.py` | Full rewrite — new 6-node Reflex architecture |
| `backend/agent.py` | 3-line debug_data fix |
| `backend/tests/unit/test_contracts.py` | New |
| `backend/tests/unit/test_adapter_schemas.py` | New |
| `backend/tests/unit/test_engine_nodes.py` | New |
| `backend/tests/unit/test_validation.py` | Deleted |
