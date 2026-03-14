# Design: Reflex Agent with LangGraph Native Tool Calling

**Date:** 2026-03-13
**Status:** Approved
**Scope:** Refactor orchestrator to use LLM-native tool calling; fix GraphRAG/Tavily root causes

---

## 1. Problem Statement

The current system has two tools (Tavily web search and GraphRAG document search) that are not being called correctly. Root causes:

1. **`.env.example` template has wrong embedding model name** — `GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-small` in the template. New deployments that copy `.env.example` inherit this wrong value. LiteLLM exposes the model as `ollama/nomic-embed-text`. GraphRAG indexing fails because the model name doesn't match any registered LiteLLM route.
2. **GraphRAG API key may be wrong class** — The `.env.example` shows a placeholder. `GRAPHRAG_API_KEY` must be the LiteLLM master key (from `litellm-config.yaml`), not a user-level key, to authorize the GraphRAG CLI subprocess calls to LiteLLM.
3. **Tool routing never triggers** — Tool selection uses a hardcoded keyword list. Most natural-language questions don't match the markers. LLM-based selection is disabled (`ORCHESTRATOR_USE_LLM_TOOL_SELECTION=false`).
4. **`max_tokens=320` causes JSON truncation** — The system prompt asks the LLM to respond in JSON with `answer`, `needs_clarification`, `confidence`, `citations`. At 320 tokens, the response is cut off, JSON is malformed, validation fails, and the system falls back.
5. **GraphRAG timeout too short** — `MCP_TOOL_TIMEOUT_MS=5000` (5s outer registry timeout) but GraphRAG CLI queries take 20-60s. The registry cancels the tool before it can complete.

---

## 2. Chosen Approach: Reflex Agent with LangGraph Native Tool Calling

**Pattern:** Single-pass Reflex agent — perceive → decide → act → respond. No retry loops.

**Core change:** Replace keyword-matching tool selection with LLM-native tool calling via `llm.bind_tools()`. The LLM reads the tool schemas at the decision node and outputs either a `tool_calls` list or a plain text response. Tool execution and final response generation remain deterministic.

**Retry loop removal:** The current `validation_node` has conditional edges back to `tool_execution_node` and `response_generation_node` for retrying. These retry edges are **removed**. Validation becomes simple: non-empty answer → END; empty/error → fallback. This is the correct Reflex behavior — one pass, no loops.

**Why not ReAct?** ReAct's iterative loops add latency and complexity. For a chatbot with two well-defined tools, a single pass is sufficient, more predictable, and easier to debug.

**Why not keyword routing?** Brittle — misses natural language variations. Requires constant maintenance as query patterns evolve. LLM tool selection is what the `tools` API is designed for.

### llama3.1 + bind_tools() Compatibility Note

llama3.1 via Ollama does support the `tools` parameter in the OpenAI-compatible `/v1/chat/completions` API. LiteLLM passes `tools` through to Ollama. However, the structured `tool_calls` output reliability varies by query complexity and model quantization. To handle this:

- **Primary path:** `llm.bind_tools(tools)` — LLM decides (no `tool_choice` parameter; `tool_choice="auto"` is NOT used because it is not reliably supported in the llama3.1/Ollama/LiteLLM chain and may cause a validation error or be silently ignored)
- **Fallback path:** If `response.tool_calls` is empty or malformed despite an apparent tool-requiring query, the system falls back to deterministic keyword routing (the current logic, kept as a secondary check)
- This means tool calling degrades gracefully: LLM-native first, keyword fallback second, direct answer last

---

## 3. Architecture

```
User message
     ↓
[classify_node]  ── greeting? ──►  instant_reply → END
     ↓ no
[tool_decision_node]
  llm.bind_tools([tavily_tool_schema, graphrag_tool_schema])  # no tool_choice param
  returns: AIMessage
     │
     ├── response.tool_calls non-empty ──► [tool_execution_node]
     │                                      registry.execute(tool_id, {query, session_id})
     │                                            ↓
     │                                     [response_generation_node]
     │                                      LLM synthesizes plain-text answer
     │                                      from tool output + message + history
     │
     ├── response.tool_calls empty,          [keyword_fallback_check]
     │   but keyword markers match ──────►   sets selected_tool deterministically
     │                                            ↓
     │                                     [tool_execution_node] → [response_generation_node]
     │
     └── no tool call, no keywords ──────► [response_generation_node]
                                            LLM answers directly from message + history
                                                  ↓
                                           [validation_node]
                                            checks: non-empty answer
                                                  │
                                                  ├── valid ──► final_response → END
                                                  └── invalid ─► [fallback_node] ──► END
```

**No retry loops.** `validation_node` does NOT have edges back to `tool_execution_node` or `response_generation_node`. If validation fails, it goes to `fallback_node` and terminates.

---

## 4. LangGraph Nodes (Complete Specification)

### 4.1 `classify_node`
- **Input state keys read:** `user_message`
- **Logic:** Call `_instant_greeting_response(user_message)`. If returns a string, set `instant_reply`. No LLM call.
- **Output state keys written:** `instant_reply: Optional[str]`
- **Routes to:** END if `instant_reply` is set; `tool_decision_node` otherwise
- **Notes:** Reuses existing `_instant_greeting_response` and `GREETING_RESPONSES` dicts unchanged

### 4.2 `tool_decision_node`
- **Input state keys read:** `user_message`, `history`, `available_tools`
- **Logic:**
  1. Build tool schemas: `available_lc_tools = [registry.get(t).adapter.to_langchain_tool() for t in available_tools]`
  2. Call `llm_with_tools = llm.bind_tools(available_lc_tools)`  — no `tool_choice` param (not reliably supported by llama3.1/Ollama)
  3. Compose `[system_message, human_message]` where system_message is a concise instruction (see system prompt in Section 4.2.1)
  4. `ai_response = await llm_with_tools.ainvoke(messages)`
  5. Parse `ai_response.tool_calls` — if non-empty, extract `tool_calls[0].name` and `tool_calls[0].args`
  6. If `tool_calls` is empty: run keyword fallback check (call `_is_project_rag_query` / `_is_web_search_query`); if matched, set `selected_tool` from keyword logic
  7. If keyword fallback also misses: `selected_tool = None`
- **Output state keys written:** `tool_call_message`, `selected_tool: Optional[str]`, `tool_args: Optional[Dict]`
- **Routes to:** `tool_execution_node` if `selected_tool` is set; `response_generation_node` otherwise
- **Error handling:** If `llm.bind_tools.ainvoke` raises an exception, log warning and proceed to `response_generation_node` with `selected_tool = None`

#### 4.2.1 Tool Decision System Prompt
```
You are a routing assistant. Given a user message, decide whether to call a tool.

Rules:
- Use `tavily_search` for: current events, news, prices, live data, web search requests
- Use `graphrag` for: questions about the uploaded thesis, project documents, or knowledge base content
- Use neither for: greetings, arithmetic, general knowledge that doesn't require retrieval

Reply only with a tool call or a brief direct answer. Do not explain your choice.
```

### 4.3 `tool_execution_node`
- **Input state keys read:** `selected_tool`, `tool_args`, `user_message`, `session_id`
- **Logic:**
  1. Determine `tool_input`:
     - If `tool_args` has a `query` key: use `{"query": tool_args["query"], "session_id": session_id}`
     - Otherwise: use `{"query": user_message, "session_id": session_id}`
  2. Call `result = await registry.execute(selected_tool, tool_input)`
  3. The registry wraps execution with timeout and retry logic (unchanged from current `registry.py`)
- **Output state keys written:** `tool_result: ToolResult`
- **Routes to:** `response_generation_node`
- **Notes:** Does NOT bypass ToolRegistry — retry/timeout wrapping in `registry.py` is preserved

### 4.4 `response_generation_node`
- **Input state keys read:** `user_message`, `history`, `tool_result` (optional), `instant_reply` (optional)
- **Logic:**
  1. If `instant_reply` is set: skip LLM call, set `final_response = instant_reply`, return
  2. Build context: `tool_context = serialize_tool_context(tool_result)` (existing helper)
  3. If `tool_result` exists and `not tool_result.get("success")`: add graceful failure message to context
  4. Compose `[system_message, human_message]` — **plain text answer requested, no JSON schema**
  5. `llm_response = await llm.ainvoke(messages)`
  6. `final_response = llm_response.content.strip()`
- **Output state keys written:** `final_response: str`, `raw_model_response: str`
- **Routes to:** `validation_node`
- **Important:** This node switches from the current JSON-envelope response format to **plain text**. This requires three coordinated changes:
  1. Replace `ASSISTANT_RESPONSE_SYSTEM_PROMPT` (currently demands JSON with 4 keys) with the plain-text prompt below
  2. `validation_node` no longer calls `_validate_response_envelope()`; checks non-empty string only
  3. `final_response` is set directly to the plain string; `ResponseEnvelope` Pydantic model is no longer used in the execution path (it can remain in `contracts.py` for backwards compatibility but is not called)
- **Why remove JSON schema?** The JSON schema requirement was causing llama3.1 to produce malformed JSON at the `max_tokens` boundary, cascading into validation failures. Plain text answers are more reliable with instruction-tuned models.

#### 4.4.1 Response Generation System Prompt (replaces `ASSISTANT_RESPONSE_SYSTEM_PROMPT`)
```
You are a helpful AI assistant. Answer the user's question clearly and concisely.
If tool output is provided, use it to inform your answer.
If the tool output indicates an error, acknowledge it gracefully without exposing technical details.
Respond in plain text. Do not use JSON format.
```

### 4.5 `validation_node`
- **Input state keys read:** `final_response`
- **Logic:** Simple check — `final_response` must be a non-empty, non-whitespace string
- **Output state keys written:** `validation_passed: bool`
- **Routes to:** END if `validation_passed`; `fallback_node` if not
- **No retry edges.** This is the key simplification. One pass, then either done or fallback.

### 4.6 `fallback_node`
- **Input state keys read:** `tool_result`, `request_id`, `retry_count`
- **Logic:**
  1. Increment `FALLBACK_EVENTS_TOTAL` metric
  2. Log fallback event
  3. Set `final_response` to a safe, user-friendly message
- **Output state keys written:** `fallback_report: FallbackReport`, `final_response: str`
- **Routes to:** END

---

## 5. Tool Schema Definition

Each adapter implements `to_langchain_tool() -> StructuredTool`. This method creates a LangChain `StructuredTool` whose `coroutine` delegates execution to `registry.execute()`, preserving timeout and retry wrapping.

**Important:** The `StructuredTool` is only used to pass the JSON schema to `bind_tools()`. Actual execution in `tool_execution_node` goes through `registry.execute()` directly (not through the StructuredTool coroutine). This preserves the registry's timeout and retry logic.

```python
# In tools/adapters.py — added to TavilyDirectAdapter
def to_langchain_tool(self) -> StructuredTool:
    class TavilyInput(BaseModel):
        query: str = Field(description="Web search query for current events, news, or facts")

    return StructuredTool(
        name="tavily_search",
        description=(
            "Search the web for current information, news, live data, prices, or recent events. "
            "Use when the user needs up-to-date information."
        ),
        args_schema=TavilyInput,
        coroutine=self.execute,  # schema only; actual calls go through registry
    )

# In tools/adapters.py — added to GraphRAGAdapter
def to_langchain_tool(self) -> StructuredTool:
    class GraphRAGInput(BaseModel):
        query: str = Field(description="Question about the uploaded thesis or document knowledge base")

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

---

## 6. GraphState — Complete New Definition

```python
class GraphState(TypedDict, total=False):
    # Identity
    request_id: str
    session_id: str

    # Input
    user_message: str
    history: List[Dict[str, str]]

    # Fast path
    instant_reply: Optional[str]          # Set by classify_node for greetings

    # Tool decision
    available_tools: List[str]
    selected_tool: Optional[str]           # Tool id chosen (or None)
    tool_args: Optional[Dict[str, Any]]    # Args from LLM tool_calls[0].args
    tool_call_message: Optional[str]       # Serialized AIMessage (str, for state compat)

    # Tool execution
    tool_result: Optional[Dict[str, Any]]  # _model_dump(ToolResult)

    # Response
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

**Fields removed vs. current (complete list):**
- `compressed_history` — history compression handled inline if needed
- `reasoning_notes` — removed; LLM handles reasoning internally
- `tool_decision` — removed; replaced by `selected_tool` (str) + `tool_args` (Dict) + `tool_call_message` (str)
- `forced_tool_attempted` — removed; no forced retry loop
- `needs_forced_tool` — removed; no forced retry loop
- `needs_regeneration` — removed; no response regeneration loop
- `max_retries` — removed; no retry budget needed (single pass, fallback on failure)
- `validation_report` — removed; validation is now a simple bool check, no detailed report needed

**Note on `tool_call_message`:** Stored as a serialized string (`json.dumps(ai_response.tool_calls)`) rather than the raw `AIMessage` object to maintain JSON serializability of the TypedDict state. The actual routing uses `selected_tool` which is already a plain string.

**Note on `tool_result`:** Stored as `Dict` (via `_model_dump`) rather than the `ToolResult` Pydantic model, consistent with how the current code handles it.

**`agent.py` impact:** `agent.py` reads `state.get("tool_decision")` for `debug_data`. This field is removed. The debug_data block in `agent.py` must be updated to read `state.get("selected_tool")` directly (it's now a top-level field) — a 3-line change.

---

## 7. Configuration Fixes

### 7.1 `.env.example` AND `.env` fixes

Both files require these changes. `env.example` is the deployment template; `.env` is the live config. Update both.

```env
# Fix 1: Embedding model name must match LiteLLM config route
GRAPHRAG_EMBEDDING_MODEL=ollama/nomic-embed-text

# Fix 2: LLM model name for GraphRAG indexing
GRAPHRAG_CHAT_MODEL=ollama/llama3.1:latest

# Fix 3: Token limit — 2048 minimum to avoid truncation
# ⚠ Apply to BOTH .env AND .env.example (current value is 320 in both files)
LITELLM_MAX_TOKENS=2048

# Fix 4: GraphRAG timeouts — outer must exceed inner by a margin
# Inner (adapter): GRAPHRAG_QUERY_TIMEOUT_SECONDS controls asyncio.wait_for inside adapter.execute()
# Outer (registry): GRAPHRAG_TIMEOUT_MS controls asyncio.wait_for in registry.execute() for graphrag tool
# Rule: outer > inner to avoid registry cancelling before adapter can return a clean error
# ⚠ GRAPHRAG_TIMEOUT_MS is a NEW variable not currently in .env or .env.example — add it
GRAPHRAG_QUERY_TIMEOUT_SECONDS=30
GRAPHRAG_TIMEOUT_MS=35000           # new variable; outer registry wrapper = 35s > inner 30s ✓
MCP_TOOL_TIMEOUT_MS=5000            # only affects Tavily (fast); GraphRAG uses GRAPHRAG_TIMEOUT_MS

# Fix 5: API key — must be LiteLLM master key (from litellm-config.yaml general_settings.master_key)
GRAPHRAG_API_KEY=sk-<your-litellm-master-key>
```

### 7.2 Why LiteLLM master key?
User-level LiteLLM API keys may have restricted model access. The GraphRAG CLI subprocess makes direct HTTP calls to the LiteLLM proxy using `GRAPHRAG_API_KEY`. If that key doesn't have access to `ollama/nomic-embed-text`, embedding calls during indexing return 403. The master key has unrestricted access.

### 7.3 `litellm-config.yaml` — Replace hardcoded master key
The current `litellm-config.yaml` contains the actual master key as a hardcoded string under `general_settings.master_key`. Before committing this file, replace the value with an environment variable reference:

```yaml
general_settings:
  master_key: ${LITELLM_MASTER_KEY}   # read from environment, not hardcoded
```

Add `LITELLM_MASTER_KEY=<your-key>` to `.env` (and use a placeholder in `.env.example`). This prevents the key from being exposed in version control.

### 7.4 `.env` (live, Docker Compose) — already correct values
`GRAPHRAG_API_BASE=http://litellm:4000` — already correct, no change needed.

---

## 8. Error Handling

| Scenario | Where caught | Behavior |
|----------|-------------|----------|
| `bind_tools()` call fails (LiteLLM/network error) | `tool_decision_node` try/except | Log warning; proceed to `response_generation_node` with `selected_tool=None` |
| LLM returns empty `tool_calls` for a tool-requiring query | `tool_decision_node` keyword fallback | Run `_is_project_rag_query` / `_is_web_search_query`; select tool if matched |
| Tool adapter returns `success=False` | `response_generation_node` checks `tool_result.get("success")` | Include graceful error message in LLM context ("tool is currently unavailable") |
| GraphRAG not ready (no index, no documents) | `readiness_error()` called in `tool_decision_node` before execution | `selected_tool` cleared; graceful message shown without tool call |
| `response_generation_node` LLM call fails | Exception caught; `final_response` set to error string | `validation_node` sees non-empty response; routes to END (error message shown) |
| `final_response` is empty or None | `validation_node` | Routes to `fallback_node` |
| `fallback_node` | Terminal | Sets safe clarification message; logs event; routes to END |

---

## 9. Files Changed

| File | Change type | Description |
|------|------------|-------------|
| `backend/orchestrator/engine.py` | Major rewrite | New node structure; `bind_tools` without `tool_choice`; keyword fallback kept as secondary path; remove retry loop edges; remove `_validate_response_envelope`; replace system prompts with plain-text versions |
| `backend/tools/adapters.py` | Additive | Add `to_langchain_tool()` method to `TavilyDirectAdapter` and `GraphRAGAdapter`; no changes to `execute()` |
| `backend/contracts.py` | Edit | Replace `GraphState` with simplified version from Section 6; `ResponseEnvelope` kept but no longer used in execution path |
| `.env.example` | Edit | Fix 5 config values (Section 7.1); add `GRAPHRAG_TIMEOUT_MS` as new variable; add `LITELLM_MASTER_KEY` placeholder |
| `.env` | Edit | Same 5 fixes as `.env.example`; update `LITELLM_MAX_TOKENS` from 320 to 2048; add `GRAPHRAG_TIMEOUT_MS=35000` |
| `litellm-config.yaml` | Security fix | Replace hardcoded `master_key` with `${LITELLM_MASTER_KEY}` env var reference |
| `backend/agent.py` | Minor edit | Update `debug_data` block to read `selected_tool` directly from state instead of via `tool_decision` |

**No changes to:** `main.py`, `registry.py`, `graphrag_service.py`, `database.py`, `cache.py`, `observability.py`, `orchestrator/context.py`, `orchestrator/json_utils.py`, `frontend/`

---

## 10. Testing Plan

1. **Config verification:** Hit `GET /tools/status`. Both `tavily_search` and `graphrag` must show `ready: true`. If `graphrag` shows a readiness error, fix the env var per Section 7 and run `POST /rag/index`.

2. **GraphRAG indexing:** `POST /rag/index` must complete without error. Check `/rag/status` shows `needs_index: false`. This confirms `GRAPHRAG_EMBEDDING_MODEL` and `GRAPHRAG_API_KEY` are correct.

3. **LLM tool calling smoke test (Tavily):** Send `POST /chat {"message": "What happened in tech news today?", "debug": true}`. Response `debug_metadata.selected_tool` must be `"tavily_search"`. Verify the answer contains web-sourced content.

4. **LLM tool calling smoke test (GraphRAG):** Send `POST /chat {"message": "What is the main contribution of the thesis?", "debug": true}`. Response `debug_metadata.selected_tool` must be `"graphrag"`. Verify the answer references document content.

5. **Direct answer (no tool):** Send `POST /chat {"message": "What is 5 times 7?"}`. No tool should be called. Response should be "35" or equivalent. Verify no tool latency in response time.

6. **Greeting fast path:** Send `POST /chat {"message": "hi"}`. Response should be instant (< 200ms). No tool called, no LLM call.

7. **Keyword fallback verification:** Temporarily modify `tool_decision_node` to force `tool_calls = []` (simulate LLM not calling tools). Send a known-keyword query like "search the web for latest AI news". Verify `selected_tool` is still set via keyword fallback.

8. **GraphRAG graceful failure:** Call `GET /tools/status` with GraphRAG marked not ready. Ask a document question. Verify the response is a graceful "knowledge base unavailable" message, not a 500 error.

---

## 11. Assumptions and Constraints

- `llama3.1` via Ollama supports the `tools` parameter through LiteLLM's OpenAI-compatible proxy. If reliability proves poor in testing, the keyword fallback (Step 7 above) ensures tool routing still works for common query patterns.
- GraphRAG indexing must be triggered manually via `POST /rag/index` after PDF upload. The bootstrap PDF path (`GRAPHRAG_BOOTSTRAP_PDF_PATH`) automates upload but not indexing.
- The LiteLLM master key in `GRAPHRAG_API_KEY` must not be rotated without also updating `.env`.
- `nomic-embed-text` must be pulled in Ollama before indexing: `ollama pull nomic-embed-text`.
- GraphRAG indexing with Ollama models is slow (5-15 minutes for a thesis-length PDF). This is a one-time operation; queries are fast after indexing.
