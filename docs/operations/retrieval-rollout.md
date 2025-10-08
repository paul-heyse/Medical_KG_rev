# Hybrid Retrieval Rollout Guide

This runbook documents the staging deployment, phased production rollout, and validation steps for the hybrid retrieval, fusion ranking, and reranking stack.

## Staging Deployment Checklist

1. **Deploy hybrid retrieval**
   - Apply the staging overlay: `kustomize build ops/k8s/overlays/staging | kubectl apply -f -`.
   - Confirms pods: `kubectl get pods -n medical-kg -l app=gateway` (should be GPU-enabled in staging).
   - Verify `configmap-retrieval` mounted at `/app/config/retrieval`.

2. **Deploy fusion ranking**
   - Ensure `MK_RERANKING__FUSION__STRATEGY` is set via the staging config map (defaults to `rrf`).
   - Run smoke tests: `k6 run tests/performance/hybrid_suite.js --vus 10 --duration 2m`.

3. **Deploy reranking service**
   - Staging patch requests 1 GPU (`nvidia.com/gpu: 1`). Confirm via `kubectl describe pod` that GPU is allocated.
   - Warm the model cache: `python scripts/run_retrieval_evaluation.py --base-url http://staging-gateway.medical-kg --rerank`.

4. **Deploy table routing**
   - Confirm intent classifier keywords via ConfigMap (see `docs/guides/retrieval/user-guide.md`).
   - Run targeted queries (`query_intent=tabular`) and verify boosted tables in response metadata.

5. **Deploy clinical boosting**
   - Validate section boosts using evaluation queries (eligibility, outcomes, dosage). Inspect `metadata.section_label`.

6. **Deploy evaluation framework**
   - CronJob `retrieval-evaluation` runs nightly in staging. Kick off manually: `kubectl create job --from=cronjob/staging-retrieval-evaluation manual-eval`.
   - Output stored in job logs and Prometheus metrics (`medicalkg_retrieval_recall_at_k`).

7. **Run staging smoke tests**
   - Execute `python tests/performance/run_retrieval_benchmarks.py --base-url http://staging-gateway ... --output staging.json`.
   - Ensure benchmark JSON shows `cache_hit_rate >= 0.4` and `per_component_latency_ms.bm25.p95 < 100`.

8. **Validate staging performance**
   - Review Grafana dashboard `Retrieval Performance` (new dashboard) for latency/nDCG metrics.
   - Confirm Prometheus alerts remain green.

## Production Rollout Phases

1. **Phase 1 – Shadow traffic (Week 1)**
   - Deploy overlay with reranking disabled (`MK_RERANKING__ENABLED=false`).
   - Mirror production queries to hybrid retrieval but continue serving legacy responses. Log results for comparison.

2. **Phase 2 – Canary (Week 2)**
   - Enable reranking for 10% of tenants via `config/retrieval/reranking.yaml` patch (production overlay sets `experiment.rerank_ratio=0.1`).
   - Monitor `retrieval_component_*` trends in Grafana and Prometheus alert `RetrievalLatencyP95High`.

3. **Phase 3 – Gradual rollout (Week 3)**
   - Increase traffic share to 50% by adjusting the experiment ratio to `0.5` and scaling gateway replicas to 4.
   - Validate A/B metrics using `python tests/performance/run_retrieval_benchmarks.py --token <prod-token> --output canary.json` and compare nDCG uplift.

4. **Phase 4 – Full rollout (Week 4)**
   - Set `default_enabled=true` for high-precision tenants (e.g. oncology) and raise canary to 100%.
   - Maintain 48-hour heightened monitoring. Alerts configured for P95 latency >600 ms or Recall@10 <75%.

5. **Table routing + clinical boosting**
   - Enable per-tenant feature flags via ConfigMap values (`MK_FEATURE_FLAGS__FLAGS__TABLE_ROUTING=true`, etc.).
   - Coordinate with domain experts to review ranked outputs.

6. **Evaluation framework**
   - Production CronJob `retrieval-evaluation` runs nightly at 02:00 UTC, writing metrics to Prometheus and optionally emitting CloudEvents.
   - Export JSON summaries to object storage by extending `scripts/run_retrieval_evaluation.py` with `--output s3://...` if required.

## Post-Deployment Validation

1. **Recall@10 improvement**
   - Use the benchmark harness to compare `recall@10` before/after rollout. Target: 65% → 82%.
   - Capture results in `/var/log/medical-kg/retrieval/rollout-metrics.json` for audit.

2. **nDCG@10 improvement**
   - Ensure `ndcg@10` increases from 0.68 → ≥0.79. Grafana panel “nDCG@10 (Evaluation)” visualises the nightly CronJob output.

3. **Latency SLA**
   - Prometheus alert `RetrievalEndToEndLatencyBreached` fires if `http_req_duration{scenario="hybrid"}` P95 exceeds 500 ms for >5 minutes. Investigate via Loki traces when triggered.

4. **User feedback**
   - Collect qualitative feedback from researchers and clinicians. Store findings in the shared Confluence page `Retrieval Rollout Q4`. Summarise top issues and action items.

5. **Deployment success report**
   - Compile a postmortem-style report covering metrics, incidents, cache hit rate trends, GPU utilisation, and recommended tweaks. Link the report in `DOCUMENTATION_UPDATES_COMPLETE.md` once finished.

## Monitoring & Alerting Summary

- **Dashboards**: `ops/monitoring/grafana/dashboards/retrieval_performance.json` provides latency, Recall@K, cache hit rate, GPU utilisation, and rerank adoption panels.
- **Alerts**:
  - `RetrievalLatencyP95High`: triggers when P95 >600 ms (warning) or >750 ms (critical).
  - `RetrievalRecallRegression`: fires if nightly evaluation drops Recall@10 by >5% from baseline.
  - `RetrievalCacheHitDrop`: warns when reranking cache hit rate falls below 40% for 15 minutes.
  - `RetrievalGpuSaturation`: warns when `reranking_gpu_utilization_percent` exceeds 90% for 10 minutes.

## Rollback Strategy

- Reapply the pre-rollout ConfigMap (`kubectl rollout undo deployment/gateway`).
- Disable CronJob: `kubectl scale cronjob retrieval-evaluation --replicas=0` if evaluation jobs overload the cluster.
- Reset feature flags via `MK_FEATURE_FLAGS__FLAGS__TABLE_ROUTING=false` etc., then redeploy.

Follow the change management process documented in `docs/operations/legacy_embedding_decommission.md` for incident handling and communication.
