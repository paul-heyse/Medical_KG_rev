# CI/CD Pipeline

The Medical KG repository ships with a comprehensive GitHub Actions pipeline that enforces code quality, contract compliance, and automated deployments.

## Workflow Overview

| Job | Purpose |
| --- | ------- |
| **Lint & Static Analysis** | Runs Black, Ruff, and MyPy against `src/` and `tests/` to ensure formatting and type safety. |
| **Unit Tests** | Executes the full pytest suite (excluding load tests) with coverage reporting. Coverage artifacts are uploaded for inspection. |
| **Integration Tests** | Boots the local Docker Compose stack (gateway, Kafka, Neo4j, OpenSearch) and runs gateway, orchestration, and service integration tests. |
| **Contract Validation** | Executes pytest contract suites, Schemathesis fuzzing, GraphQL Inspector diff, and Buf breaking-change detection against `origin/main`. |
| **Performance Tests** | Nightly/dispatch job executing k6 scenarios (`retrieve_latency`, `ingest_throughput`, `concurrency`) with latency thresholds. |
| **Docker Image** | Builds and pushes `ghcr.io/<org>/medical-kg` images tagged with the commit SHA and `latest`. |
| **Deploy Staging** | Applies the Kubernetes overlay in `ops/k8s/overlays/staging` automatically for commits on `main`. |
| **Deploy Production** | Requires manual approval before applying the production overlay; runs only on `main`. |
| **Docs** | Builds the MkDocs site and publishes to GitHub Pages. |
| **Branch Protection** | Optional workflow-dispatch job that enforces protection rules using `peter-evans/create-or-update-branch-protection`. |

## Required Secrets

| Secret | Description |
| ------ | ----------- |
| `GITHUB_TOKEN` | Provided automatically; used for pushing container images and docs. |
| `ADMIN_TOKEN` | Personal access token with `repo` and `admin:repo_hook` scope for branch protection automation. |
| `KUBE_CONFIG` *(optional)* | When supplied, enables direct `kubectl` access during deployment steps. |

## Running Locally

```bash
poetry install --with dev
pytest
pytest tests/contract
schemathesis run --app=Medical_KG_rev.gateway.app:create_app docs/openapi.yaml
```

Use `docker compose -f ops/docker-compose.yml up` to reproduce the integration environment before running tests.
