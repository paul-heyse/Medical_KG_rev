# Local Environments

The repository provides a fully integrated Docker Compose stack for development and QA located at `ops/docker-compose.yml`.

## Services

| Service | Purpose |
| ------- | ------- |
| `gateway` | FastAPI multi-protocol gateway with metrics, tracing, and Sentry hooks. |
| `kafka` / `zookeeper` | Event bus for ingestion orchestration. |
| `neo4j` | Knowledge graph backing store. |
| `opensearch` | Vector + document search index. |
| `jaeger` | Trace collection and visualization. |
| `prometheus` | Metrics storage. |
| `grafana` | Dashboarding. |
| `loki` / `promtail` | Structured log aggregation. |

## Usage

```bash
cp .env.example .env
docker compose -f ops/docker-compose.yml up --build
# Access services:
# Gateway: http://localhost:8000
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686
```

Use `docker compose down -v` to stop the stack and remove volumes.

## Health Checks

Each service defines `healthcheck` commands. The helper script `scripts/wait_for_services.sh` waits until the gateway exposes `/openapi.json`, ensuring readiness for integration tests.
