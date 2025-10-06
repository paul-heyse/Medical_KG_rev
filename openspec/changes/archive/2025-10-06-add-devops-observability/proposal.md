# Change Proposal: DevOps & Observability

## Why

Establish production-ready deployment infrastructure with CI/CD pipelines, comprehensive observability (Prometheus metrics, OpenTelemetry tracing, Grafana dashboards), contract/performance tests (Schemathesis, k6), containerization (Docker Compose + Kubernetes), and developer tooling.

## What Changes

- CI/CD pipeline (GitHub Actions) with lint, test, build, deploy stages
- Contract tests (Schemathesis for OpenAPI, GraphQL Inspector, Buf for gRPC)
- Performance tests (k6) with P95 latency assertions
- Prometheus metrics exposition and scraping
- OpenTelemetry distributed tracing (Jaeger)
- Grafana dashboards for system health, GPU utilization, API latencies
- Structured logging with correlation IDs
- Error tracking (Sentry integration)
- Docker Compose for local development
- Kubernetes manifests (deployments, services, ingress, HPA)
- Helm chart (optional)
- Monitoring alerts (error rate, latency spikes, GPU saturation)
- Documentation site (MkDocs) with API docs

## Impact

- **Affected specs**: NEW capability `devops-observability`
- **Affected code**:
  - `.github/workflows/` - CI/CD pipelines
  - `ops/docker-compose.yml` - Local development stack
  - `ops/k8s/` - Kubernetes manifests
  - `ops/monitoring/prometheus.yml` - Prometheus config
  - `ops/monitoring/grafana/` - Grafana dashboards
  - `tests/contract/` - Schemathesis tests
  - `tests/performance/` - k6 load tests
  - `docs/` - MkDocs documentation site
