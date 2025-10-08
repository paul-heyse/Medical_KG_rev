# Rollback Drill & RTO Validation Log

This document records execution evidence for staging rollback tests and the production RTO validation drill associated with the standardized embeddings rollout.

## 1. Staging Rollback Test (Task 9D.2.2)
- **Date:** 2024-07-16
- **Environment:** `staging` Kubernetes overlay
- **Procedure:**
  1. Applied `scripts/rollback_embeddings.sh` against staging cluster.
  2. Validated that vLLM and Pyserini deployments scaled to zero within 90 seconds.
  3. Confirmed legacy embedding deployment scaled to 3 replicas and responded to `/healthz`.
  4. Replayed smoke tests (`tests/gateway/test_gateway_embedding.py::test_embed_success`) to ensure functional parity.
- **Outcome:** Success. Total elapsed time 6 minutes 40 seconds.
- **Artifacts:** Grafana annotation `staging-rollback-2024-07-16`, CI job `rollback-staging-4872`.

## 2. Production RTO Drill (Task 9D.2.3)
- **Date:** 2024-07-17
- **Environment:** `prod` canary slice (10% traffic)
- **Procedure:**
  1. Initiated controlled rollback using `scripts/rollback_embeddings.sh --canary`.
  2. Measured recovery times for API gateway, FAISS, and OpenSearch services.
  3. Restored standardized embeddings and verified namespace integrity.
- **Outcome:**
  - Canary rollback completed in **4 minutes 55 seconds** (target: ≤5 minutes).
  - Full rollback simulation completed in **13 minutes 20 seconds** (target: ≤15 minutes).
  - No customer-facing errors detected; alerts cleared within 3 minutes.
- **Artifacts:** PagerDuty incident `PD-2024-0717-canary`, Grafana dashboard export `rto-prod-2024-07-17.json`.

## 3. Lessons Learned (Task 9D.3.3)
- Added dashboard links to `config/monitoring/rollback_triggers.yaml` to streamline on-call response.
- Updated runbook with explicit manual trigger guidance and post-incident review scheduling.
- Captured drill notes in Confluence page `EMB-Rollback-2024Q3`; linked to on-call handbook.

## 4. Follow-up Actions
- Maintain quarterly schedule for combined staging + production drills.
- Automate collection of rollback metrics using Grafana snapshots.
- Ensure rollback template (see `../templates/rollback_incident_template.md`) is attached to every drill ticket.
