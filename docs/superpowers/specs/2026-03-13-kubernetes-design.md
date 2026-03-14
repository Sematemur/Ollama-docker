# Kubernetes Architecture Design

## Overview

Migration from Docker Compose to Kubernetes for the chat system. Separates FastAPI (API) and Agent (LangGraph orchestration) into independent microservices communicating via HTTP/REST. Single-node cluster with resource management, secrets, configmaps, and a full observability stack.

## Architecture

### Service Separation

The monolithic backend is split into two microservices:

- **api-service** (port 8000): FastAPI app handling HTTP requests, chat history (PostgreSQL), response caching (Redis). Calls agent-service via internal HTTP.
- **agent-service** (port 8001): Standalone FastAPI app running LangGraph orchestration and LLM calls. Exposed only within the cluster via ClusterIP.

Communication: `api-service → HTTP POST /invoke → agent-service`

### Workload Types

| Workload | Kind | Replicas | Reason |
|---|---|---|---|
| PostgreSQL | StatefulSet + PVC (5Gi) | 1 | Stateful, persistent storage required |
| Redis | Deployment (no PVC) | 1 | Used as cache, data loss acceptable |
| api-service | Deployment | 2 | Stateless, horizontally scalable |
| agent-service | Deployment | 2 | Stateless, independently scalable |
| Frontend | Deployment | 1 | Stateless |
| LiteLLM | Deployment | 1 | Stateless proxy |
| Prometheus | Deployment + PVC (5Gi) | 1 | Metrics storage |
| Loki | Deployment + PVC (5Gi) | 1 | Log storage |
| Tempo | Deployment + PVC (5Gi) | 1 | Trace storage |
| Grafana | Deployment + PVC (2Gi) | 1 | Dashboard state |

### Networking

| Service | Type | Port Mapping |
|---|---|---|
| postgres-service | ClusterIP | 5432 |
| redis-service | ClusterIP | 6379 |
| agent-service | ClusterIP | 8001 |
| litellm-service | ClusterIP | 4000 |
| prometheus-service | ClusterIP | 9090 |
| loki-service | ClusterIP | 3100 |
| tempo-service | ClusterIP | 3200, 4317 |
| api-service | NodePort | 8000 → 30080 |
| frontend-service | NodePort | 5173 → 30173 |
| grafana-service | NodePort | 3000 → 30300 |

### Configuration Management

- **ConfigMap** (`chat-config`): model name, cache TTL, history limits, CORS origins, OTEL endpoints, agent service URL, all non-sensitive settings.
- **Secret** (`chat-secrets`): PostgreSQL password, Redis password, LiteLLM API key/master key, LangChain API key, Tavily API key, Grafana credentials.
- **LiteLLM ConfigMap** (`litellm-config`): LiteLLM model routing configuration.
- **Observability ConfigMaps**: Prometheus, Loki, Tempo configuration files.

### Resource Limits

| Service | CPU Request/Limit | Memory Request/Limit |
|---|---|---|
| PostgreSQL | 250m / 500m | 256Mi / 512Mi |
| Redis | 100m / 250m | 128Mi / 256Mi |
| api-service | 250m / 500m | 256Mi / 512Mi |
| agent-service | 500m / 1000m | 512Mi / 1Gi |
| Frontend | 100m / 250m | 128Mi / 256Mi |
| LiteLLM | 250m / 500m | 256Mi / 512Mi |
| Observability (each) | 100m / 250m | 128-256Mi / 256-512Mi |

### Health Checks

All services have liveness and readiness probes:
- API/Agent: `GET /health`
- PostgreSQL: `pg_isready`
- Redis: `redis-cli ping`
- LiteLLM: `GET /health/liveliness`
- Prometheus: `GET /-/healthy`
- Loki/Tempo: `GET /ready`
- Grafana: `GET /api/health`

### Debugging

- `debug-pod`: nicolaka/netshoot image with curl, nslookup, psql, redis-cli, tcpdump
- Secrets injected as env vars for database/cache testing
- Usage examples documented in the manifest

## Code Changes

1. **New `backend/agent_service.py`**: Standalone FastAPI app exposing `POST /invoke` and `GET /health`
2. **New `backend/Dockerfile.agent`**: Dockerfile for agent service (port 8001)
3. **Modified `backend/main.py`**: Uses `httpx.AsyncClient` to call agent-service via HTTP instead of direct import
4. **Modified `backend/config.py`**: Added `agent_service_url` setting

## File Structure

```
k8s/
├── namespace.yaml
├── configmap.yaml
├── secrets.yaml
├── postgres/
│   ├── statefulset.yaml
│   └── service.yaml
├── redis/
│   ├── deployment.yaml
│   └── service.yaml
├── api/
│   ├── deployment.yaml
│   └── service.yaml
├── agent/
│   ├── deployment.yaml
│   └── service.yaml
├── frontend/
│   ├── deployment.yaml
│   └── service.yaml
├── litellm/
│   ├── configmap.yaml
│   ├── deployment.yaml
│   └── service.yaml
├── observability/
│   ├── prometheus/
│   │   ├── configmap.yaml
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── loki/
│   │   ├── configmap.yaml
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── tempo/
│   │   ├── configmap.yaml
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── grafana/
│       ├── deployment.yaml
│       └── service.yaml
└── debug/
    └── debug-pod.yaml
```

## Deployment Order

```bash
# 1. Namespace
kubectl apply -f k8s/namespace.yaml

# 2. Configuration
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# 3. Data layer
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/redis/

# 4. LLM proxy
kubectl apply -f k8s/litellm/

# 5. Application services
kubectl apply -f k8s/agent/
kubectl apply -f k8s/api/
kubectl apply -f k8s/frontend/

# 6. Observability
kubectl apply -f k8s/observability/prometheus/
kubectl apply -f k8s/observability/loki/
kubectl apply -f k8s/observability/tempo/
kubectl apply -f k8s/observability/grafana/

# 7. Debug (optional)
kubectl apply -f k8s/debug/
```
