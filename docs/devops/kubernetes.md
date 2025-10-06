# Kubernetes Deployment

Kustomize manifests under `ops/k8s/` support both staging and production environments.

## Base Resources

- **Namespace** – `medical-kg` isolates all gateway workloads.
- **ConfigMap / Secret** – Provide runtime configuration and Sentry DSN injection.
- **Deployments** – `gateway` (FastAPI) and `ingest-worker` (background processing).
- **Service & Ingress** – Expose the HTTP gateway via `medical-kg.local` (override host in overlays).
- **HorizontalPodAutoscaler** – Scales the gateway between 2 and 6 replicas targeting 70% CPU.

## Overlays

| Overlay | Purpose |
| ------- | ------- |
| `staging` | Single replica deployments with staging container tags. Useful for QA clusters. |
| `production` | Multi-replica deployments using the `latest` image tag and higher replica counts. |

## Deployment Workflow

```bash
# Staging
kubectl apply -k ops/k8s/overlays/staging

# Production (requires manual approval in CI)
kubectl apply -k ops/k8s/overlays/production
```

Update `ops/k8s/base/configmap-gateway.yaml` for telemetry endpoints (Jaeger, OTLP) and supply secrets via the `gateway-sentry` secret or an external secret manager.
