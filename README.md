# Chat Application

A self-hosted chat application powered by local LLMs (Ollama). Your data stays on your own infrastructure.

## Architecture

The system follows a **microservices architecture** with two deployment modes: Docker Compose (local development) and Kubernetes (production).

```
┌───────────┐     ┌───────────┐     ┌───────────────┐     ┌───────────┐
│  Frontend │────>│  API      │────>│ Agent Service │────>│  LiteLLM  │──> Ollama
│  (React)  │     │ (FastAPI) │     │  (LangGraph)  │     │ (Gateway) │
│  :5173    │     │  :8000    │     │   :8001       │     │  :4000    │
└───────────┘     └─────┬─────┘     └───────┬───────┘     └───────────┘
                        │                   │
                  ┌─────┴─────┐       ┌─────┴─────┐
                  │ PostgreSQL│       │   Redis   │
                  │  :5432    │       │  :6379    │
                  └───────────┘       └───────────┘
```

### Services

| Service | Purpose | Port |
|---------|---------|------|
| **Frontend** | React (Vite) chat UI | 5173 |
| **API** | FastAPI — routes, auth, history, cache | 8000 |
| **Agent Service** | LangGraph orchestration, LLM calls, tool execution | 8001 |
| **LiteLLM** | Unified LLM gateway to Ollama | 4000 |
| **PostgreSQL** | Chat history persistence | 5432 |
| **Redis** | Response cache with TTL | 6379 |
| **Prometheus** | Metrics collection | 9090 |
| **Loki** | Log aggregation | 3100 |
| **Tempo** | Distributed tracing | 4317 |
| **Grafana** | Dashboards and visualization | 3000 |

## Prerequisites

- **Docker Desktop** with Docker Compose
- **Ollama** running locally (`ollama serve`) with models pulled:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull llama3.1:latest
  ```
- **minikube** (for Kubernetes deployment)

## Quick Start — Docker Compose

1. Copy and configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

2. Start all services:
   ```bash
   docker compose up --build
   ```

3. Open the chat UI at **http://localhost:5173**

4. Stop services:
   ```bash
   docker compose down
   ```

## Kubernetes Deployment (minikube)

### Initial Setup

1. Start minikube:
   ```bash
   minikube start
   ```

2. Point your shell to minikube's Docker daemon and build the images:
   ```bash
   eval $(minikube docker-env)
   docker build -t chat-api:latest -f backend/Dockerfile ./backend
   docker build -t chat-agent:latest -f backend/Dockerfile.agent ./backend
   docker build -t chat-frontend:latest ./frontend
   ```

3. Update secrets with your actual values:
   ```bash
   # Edit k8s/secrets.yaml with real API keys and passwords
   ```

4. Apply manifests in order:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/secrets.yaml
   kubectl apply -f k8s/configmap.yaml

   # Infrastructure
   kubectl apply -f k8s/postgres/
   kubectl apply -f k8s/redis/
   kubectl apply -f k8s/litellm/

   # Observability
   kubectl apply -f k8s/observability/prometheus/
   kubectl apply -f k8s/observability/loki/
   kubectl apply -f k8s/observability/tempo/
   kubectl apply -f k8s/observability/grafana/

   # Application
   kubectl apply -f k8s/agent/
   kubectl apply -f k8s/api/
   kubectl apply -f k8s/frontend/
   ```

5. Wait for all pods to be ready:
   ```bash
   kubectl get pods -n chat-system -w
   ```

### Accessing Services

On Windows with Docker driver, use `kubectl port-forward`:

```bash
# If Docker Compose is also running, use alternative ports (9000/9173)
kubectl port-forward svc/api-service 9000:8000 -n chat-system &
kubectl port-forward svc/frontend-service 9173:5173 -n chat-system &
```

Then open the chat UI at **http://localhost:9173**

### K8s Architecture Details

```
Namespace: chat-system

┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Deployments (auto-restart on crash):                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ API (x2) │  │Agent (x2)│  │Frontend  │              │
│  │ :8000    │  │ :8001    │  │ :5173    │              │
│  └────┬─────┘  └──────────┘  └──────────┘              │
│       │                                                 │
│  StatefulSet:         Deployments:                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │PostgreSQL│  │  Redis   │  │ LiteLLM  │              │
│  │  (PVC)   │  │          │  │          │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                         │
│  Observability:                                         │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │
│  │Promeths│ │  Loki  │ │ Tempo  │ │Grafana │          │
│  └────────┘ └────────┘ └────────┘ └────────┘          │
└─────────────────────────────────────────────────────────┘
```

**Resilience features:**
- All Deployments have `restartPolicy: Always` — crashed pods are automatically restarted by kubelet
- API and Agent run **2 replicas** each for high availability
- PostgreSQL uses a **StatefulSet with PVC** (5Gi) for data persistence across restarts
- API pods use an **init container** that waits for PostgreSQL readiness before starting
- Database tables are auto-created on application startup via `init_db()` — no manual SQL needed
- Liveness and readiness probes configured on all services
- Secrets are managed via Kubernetes Secrets (separate from ConfigMap)

**Exposed services (NodePort):**

| Service | NodePort |
|---------|----------|
| API | 30080 |
| Frontend | 30173 |
| Grafana | 30300 |

## Known Limitations

- The default model (`ollama/qwen2.5:3b`) is a lightweight 3B-parameter model optimized for speed over accuracy. It may produce **hallucinated or repetitive responses**, especially when tool output is incomplete or ambiguous (e.g., repeating the same weather data for 7 days when only current data is available).
- For better response quality, consider switching to a larger model by updating `LITELLM_MODEL_NAME` in your `.env` or ConfigMap.

## LLM Response Cache (Redis)

A Redis-based response cache avoids repeated LLM calls for the same question.

**Flow:**
1. Normalize question (trim, lowercase, collapse whitespace)
2. Build cache key from `sha256(normalized_question)`
3. Check Redis — if found, return cached response
4. If missing, call LLM and write response to Redis with TTL

**Environment variables:**
- `REDIS_URL` — connection string (default: `redis://localhost:6379/0`)
- `REDIS_CACHE_TTL_SECONDS` — cache expiry (default: `600` = 10 minutes)
- `ENABLE_RESPONSE_CACHE` — toggle (`true` / `false`)
- `CACHE_SENSITIVE_PATTERNS` — skip caching responses containing secret markers

**Infrastructure:**
- Redis persistence: snapshot + AOF (`--save 60 1 --appendonly yes`)
- Password-protected via `REDIS_PASSWORD`
- Connection pooling via `REDIS_MAX_CONNECTIONS`

## MCP Reflex Agent (LangGraph)

The backend uses a **single-agent reflex graph** with the following nodes:

```
START → classify → tool_decision → tool_execution → response_generation → validation → END
                                                                              ↓
                                                                          fallback → END
```

### Tool Architecture

- Tools are integrated through **adapter + registry** pattern (`backend/tools/`)
- Tool list is dynamic via `MCP_ENABLED_TOOLS` environment variable
- Available tools:
  - `tavily_search` — web search via Tavily API

### Validation / Retry / Fallback

- Tool call retries use **exponential backoff**
- Keyword-based fallback routing when LLM returns no `tool_calls`
- If generation fails, system returns a safe clarification message (no model switch)
- Anti-hallucination prompt: LLM is instructed to only use facts from tool output and admit when data is insufficient

### Observability Stack

| Component | Purpose |
|-----------|---------|
| **LangSmith** | Prompt and chain traces |
| **Prometheus** | Metrics (`GET /metrics`) |
| **Loki** | Centralized log aggregation |
| **Tempo** | Distributed tracing (OpenTelemetry) |
| **Grafana** | Unified dashboards |

### Evaluation Framework

Run evaluations against datasets (`.jsonl` or `.csv`):

```bash
# CLI
python -m backend.eval.run --dataset path/to/eval.jsonl

# API
POST /eval/run
{"dataset_path": "..."}
```

Dataset fields: `query`, `expected_tool`, `must_use_tool`, `notes`

## Environment Variables

All configuration is managed via environment variables. See `.env.example` for the complete list with descriptions and default values.

## Project Structure

```
├── backend/
│   ├── main.py              # API service (FastAPI)
│   ├── agent_service.py     # Agent microservice (FastAPI)
│   ├── agent.py             # Orchestrator setup (LLM + registry)
│   ├── config.py            # Pydantic BaseSettings
│   ├── database.py          # PostgreSQL helpers
│   ├── cache.py             # Redis response cache
│   ├── observability.py     # Logging, metrics, tracing
│   ├── contracts.py         # Shared Pydantic models
│   ├── orchestrator/
│   │   └── engine.py        # LangGraph orchestration graph
│   ├── tools/
│   │   ├── adapters.py      # Tool adapters (Tavily)
│   │   └── registry.py      # Tool registry with retry logic
│   ├── eval/
│   │   └── run.py           # Evaluation runner
│   ├── Dockerfile           # API service image
│   └── Dockerfile.agent     # Agent service image
├── frontend/
│   └── src/components/
│       └── Chat.jsx         # Chat UI component
├── k8s/                     # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── api/                 # API deployment + service
│   ├── agent/               # Agent deployment + service
│   ├── frontend/            # Frontend deployment + service
│   ├── postgres/            # PostgreSQL StatefulSet
│   ├── redis/               # Redis deployment
│   ├── litellm/             # LiteLLM deployment
│   └── observability/       # Prometheus, Loki, Tempo, Grafana
├── observability/           # Docker Compose observability configs
├── docker-compose.yml       # Local development stack
├── litellm-config.yaml      # LiteLLM proxy configuration
└── .env.example             # Environment variable template
```
