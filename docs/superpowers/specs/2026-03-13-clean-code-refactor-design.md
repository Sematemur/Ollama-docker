# Clean Code Refactor Design

**Date:** 2026-03-13
**Scope:** Full stack — backend (Python/FastAPI) + frontend (React)

---

## Goal

Refactor the codebase to clean code principles: no hardcoded values, no redundant code, proper logging, modular config, and production-ready patterns. Remove all GraphRAG functionality.

---

## Section 1: GraphRAG Removal

**Files deleted:**
- `backend/graphrag_service.py`

**Code removed from existing files:**
- `main.py`: `GRAPHRAG_MANAGER`, `bootstrap_seed_graphrag_workspace()`, `/rag/upload`, `/rag/index`, `/rag/status` endpoints, graphrag import, graphrag references in `run_eval`
- `orchestrator/engine.py`: `PROJECT_RAG_MARKERS`, `_is_project_rag_query()`, graphrag branch in `tool_decision_node`, graphrag case in `_graceful_tool_failure_message()`
- `tools/adapters.py`: `GraphRAGAdapter` class, import of `GraphRAGWorkspaceManager`
- `.env.example`: all `GRAPHRAG_*` variables

---

## Section 2: Central `config.py`

New `backend/config.py` using Pydantic `BaseSettings`. All env vars are defined once with types and defaults. All `os.getenv()` calls across backend files are replaced with `settings.<field>`.

**Fields:**
- LiteLLM: `litellm_api_key`, `litellm_base_url`, `litellm_model_name`, `litellm_request_timeout`, `litellm_temperature`, `litellm_max_tokens`, `litellm_max_retries`
- PostgreSQL: `database_url`
- Redis: `redis_url`, `redis_cache_ttl_seconds`, `redis_max_connections`, `enable_response_cache`, `cache_sensitive_patterns`
- Chat/Orchestrator: `chat_history_max_turns`, `orchestrator_history_turns`, `orchestrator_tool_context_max_chars`
- Tools: `tavily_api_key`, `tavily_topic`, `tavily_search_depth`, `tavily_max_results`, `tavily_timeout_seconds`, `tavily_max_retries`, `mcp_tool_timeout_ms`, `mcp_tool_max_retries`, `mcp_tool_initial_backoff_ms`, `mcp_enabled_tools`
- Observability: `otel_service_name`, `otel_exporter_otlp_endpoint`, `loki_push_url`, `service_version`

`pydantic-settings` added to `requirements.txt`.

---

## Section 3: Logging, Deprecated APIs, Hardcoded Strings

- `database.py`: `print()` → `logging.getLogger("chat.database")`
- `cache.py`: all `print()` calls → `logging.getLogger("chat.cache")`
- `main.py`: migrate `@app.on_event("startup/shutdown")` → `lifespan` context manager
- `orchestrator/engine.py`: repeated `os.getenv("ORCHESTRATOR_HISTORY_TURNS")` → module-level constant from `settings`; fallback/error strings → named constants (`_FALLBACK_MESSAGE`, `_ERROR_RESPONSE_MESSAGE`, `_NO_RESPONSE_MESSAGE`)
- `observability.py`: hardcoded `"service.version": "1.0.0"` → `settings.service_version`
- `main.py` CORS: remove redundant `"*"` wildcard; keep only explicit origins (`http://localhost:5173`, `http://localhost:3000`) configurable via `settings`

---

## Section 4: Shared Redis Helper + Frontend Cleanup

**`tools/adapters.py`:**
Extract duplicated lazy Redis init into a module-level `_RedisCache` helper class. `TavilyDirectAdapter` uses `_tool_redis.get_client()`. All inline `os.getenv()` calls for Redis config → `settings`.

**`frontend/Chat.jsx`:**
- Remove all tutorial-style Turkish block comments and section headers
- Remove `console.log()` debug call
- Replace deprecated `onKeyPress` → `onKeyDown`
- Keep inline `styles` object (no CSS extraction)

---

## Files Changed Summary

| File | Action |
|---|---|
| `backend/config.py` | CREATE |
| `backend/graphrag_service.py` | DELETE |
| `backend/main.py` | MODIFY |
| `backend/agent.py` | MODIFY |
| `backend/database.py` | MODIFY |
| `backend/cache.py` | MODIFY |
| `backend/observability.py` | MODIFY |
| `backend/orchestrator/engine.py` | MODIFY |
| `backend/tools/adapters.py` | MODIFY |
| `backend/requirements.txt` | MODIFY |
| `.env.example` | MODIFY |
| `frontend/src/components/Chat.jsx` | MODIFY |
