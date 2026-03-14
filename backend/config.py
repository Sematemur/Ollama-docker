"""Centralised application configuration via Pydantic BaseSettings.

All environment variables are declared here with their types and defaults.
Import ``settings`` from this module instead of calling ``os.getenv`` directly.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # LiteLLM / LLM
    # ------------------------------------------------------------------
    litellm_api_key: str = ""
    litellm_base_url: str = "http://localhost:8023/"
    litellm_model_name: str = "ollama/qwen2.5:3b"
    litellm_request_timeout: int = 45
    litellm_temperature: float = 0.2
    litellm_max_tokens: int = 2048
    litellm_max_retries: int = 1

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    database_url: str = "postgresql://chatuser:chatpass@localhost:5432/chatdb"

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = 600
    redis_max_connections: int = 50
    enable_response_cache: bool = True
    cache_sensitive_patterns: str = (
        "sk-,api_key,password,secret,token,authorization,bearer"
    )

    # ------------------------------------------------------------------
    # Chat / Orchestrator
    # ------------------------------------------------------------------
    chat_history_max_turns: int = 12
    orchestrator_history_turns: int = 6
    orchestrator_tool_context_max_chars: int = 2500

    # ------------------------------------------------------------------
    # Tavily
    # ------------------------------------------------------------------
    tavily_api_key: str = ""
    tavily_topic: str = "general"
    tavily_search_depth: str = "basic"
    tavily_max_results: int = 3
    tavily_timeout_seconds: float = 8.0
    tavily_max_retries: int = 1

    # ------------------------------------------------------------------
    # MCP / Tool registry
    # ------------------------------------------------------------------
    mcp_enabled_tools: str = "tavily_search"
    mcp_tool_timeout_ms: int = 5000
    mcp_tool_max_retries: int = 1
    mcp_tool_initial_backoff_ms: int = 150

    # ------------------------------------------------------------------
    # Agent service (for API → Agent HTTP calls in K8s)
    # ------------------------------------------------------------------
    agent_service_url: str = "http://localhost:8001"

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    otel_service_name: str = "chat-backend"
    otel_exporter_otlp_endpoint: str = ""
    loki_push_url: str = ""
    service_version: str = "2.0.0"

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


settings = Settings()
