# Operations Manual: MinerU vLLM Split-Container Architecture

**Change ID**: `update-mineru-vllm-split-container`
**Version**: 1.0
**Date**: 2025-10-08
**Status**: Operational

---

## Table of Contents

1. [Service Level Objectives (SLOs)](#service-level-objectives-slos)
2. [Incident Response Runbook](#incident-response-runbook)
3. [Capacity Planning](#capacity-planning)
4. [Maintenance Procedures](#maintenance-procedures)
5. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
6. [On-Call Procedures](#on-call-procedures)
7. [Performance Tuning](#performance-tuning)

---

## Service Level Objectives (SLOs)

### SLO Definitions

#### SLO 1: Availability

**Definition**: Percentage of time MinerU PDF processing service is available and operational

**Target**: **99.5%** availability (monthly)

**Measurement**:

```promql
# Uptime = successful health checks / total health checks
(
  count_over_time(up{job="vllm-server"}[30d])
  -
  count_over_time((up{job="vllm-server"} == 0)[30d])
)
/
count_over_time(up{job="vllm-server"}[30d])
```

**Acceptable Downtime**:

- **Monthly**: 3.6 hours (30 days × 0.5% = 3.6h)
- **Weekly**: 50 minutes
- **Daily**: 7 minutes

**Exclusions**: Planned maintenance windows (announced 48h in advance)

**Alerting**: Page on-call if availability drops below 99.5% projected for the month

---

#### SLO 2: Processing Latency

**Definition**: Time to process a single PDF from worker intake to output storage

**Targets**:

- **P50**: <45 seconds
- **P95**: <90 seconds
- **P99**: <120 seconds

**Measurement**:

```promql
# P95 latency over 5-minute window
histogram_quantile(0.95,
  rate(mineru_pdf_processing_duration_seconds_bucket[5m])
)
```

**Alerting**: Page on-call if P95 > 90s for 15 minutes

---

#### SLO 3: Throughput

**Definition**: Number of PDFs successfully processed per hour

**Target**: **≥60 PDFs/hour** (baseline: 50 PDFs/hour, +20% improvement)

**Measurement**:

```promql
# PDFs processed per hour
rate(mineru_pdfs_processed_total[1h]) * 3600
```

**Alerting**: Page on-call if throughput < 50 PDFs/hour for 30 minutes

---

#### SLO 4: Error Rate

**Definition**: Percentage of PDF processing jobs that fail

**Target**: **<2%** error rate

**Measurement**:

```promql
# Error rate = failed jobs / total jobs
(
  rate(mineru_pdfs_failed_total[5m])
  /
  rate(mineru_pdfs_processed_total[5m] + mineru_pdfs_failed_total[5m])
) * 100
```

**Alerting**: Page on-call if error rate > 5% for 10 minutes

---

#### SLO 5: vLLM Server Uptime

**Definition**: Percentage of time vLLM inference server is healthy and responsive

**Target**: **99.9%** uptime (monthly)

**Measurement**:

```promql
# vLLM server uptime
(
  count_over_time(up{job="vllm-server"}[30d])
  -
  count_over_time((up{job="vllm-server"} == 0)[30d])
)
/
count_over_time(up{job="vllm-server"}[30d])
```

**Acceptable Downtime**: 43 minutes/month

**Alerting**: Page on-call immediately if vLLM server down for >5 minutes

---

### SLO Dashboard (Grafana)

**Dashboard**: `MinerU Split-Container SLOs`

**Panels**:

1. **Availability**: 30-day rolling window, current vs target (99.5%)
2. **Latency**: P50/P95/P99 time series, threshold lines
3. **Throughput**: PDFs/hour, 7-day moving average
4. **Error Rate**: Percentage, color-coded (green <2%, yellow 2-5%, red >5%)
5. **vLLM Uptime**: 30-day uptime, incidents annotated

**URL**: `https://grafana.example.com/d/mineru-slos`

---

## Incident Response Runbook

### Incident Classification

| Severity | Criteria | Response Time | Escalation |
|----------|----------|---------------|------------|
| **Sev1** | Service down, data loss | Immediate | Page on-call + Engineering Lead |
| **Sev2** | Degraded performance (SLO breach) | <15 min | Page on-call |
| **Sev3** | Minor issues, no user impact | <1 hour | Slack notification |

---

### Runbook 1: vLLM Server Down

**Symptoms**:

- Alert: `vLLM server down for >5 minutes`
- Workers logging `ConnectionRefused` or `CircuitBreakerOpen`
- Throughput drops to 0

**Diagnosis**:

```bash
# Check vLLM server pod status
kubectl get pods -l app=vllm-server -n medical-kg

# Check logs
kubectl logs -l app=vllm-server -n medical-kg --tail=100

# Common failure modes:
# - OOM killed (check: kubectl describe pod)
# - GPU driver crash (check: nvidia-smi in pod)
# - Model loading failure (check logs for "Failed to load model")
```

**Resolution**:

**Case 1: Pod OOMKilled**

```bash
# Cause: GPU memory exhausted
# Solution: Restart pod (Kubernetes does automatically)

# Validate restart successful
kubectl get pods -l app=vllm-server -n medical-kg
# Wait for STATUS: Running, READY: 1/1 (may take 60s for model loading)

# Check GPU memory
kubectl exec -it vllm-server-xxx -n medical-kg -- nvidia-smi

# If recurring: Reduce batch size or max_model_len
kubectl set env deployment/vllm-server GPU_MEMORY_UTILIZATION=0.85 -n medical-kg
```

**Case 2: GPU Driver Crash**

```bash
# Symptom: Pod stuck in CrashLoopBackOff, logs show "CUDA error"
# Solution: Cordon node, drain workloads, restart node

# Cordon node (prevent new pods)
kubectl cordon <node-name>

# Drain vLLM pod
kubectl delete pod vllm-server-xxx -n medical-kg

# vLLM will reschedule to another GPU node
# If no other GPU node: Reboot affected node
```

**Case 3: Model Loading Failure**

```bash
# Symptom: Logs show "Failed to load model from Hugging Face"
# Cause: Network issue or invalid HF_TOKEN

# Check HuggingFace connectivity
kubectl exec -it vllm-server-xxx -n medical-kg -- curl -I https://huggingface.co

# Check HF_TOKEN secret
kubectl get secret vllm-secrets -n medical-kg -o yaml

# If token expired: Regenerate and update secret
kubectl create secret generic vllm-secrets \
  --from-literal=hf-token=$NEW_HF_TOKEN \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart vLLM server
kubectl rollout restart deployment/vllm-server -n medical-kg
```

**Post-Incident**:

- Update incident log
- Calculate downtime impact on SLO
- Schedule post-mortem if Sev1

**Time to Resolution**: 5-15 minutes

---

### Runbook 2: High Latency (P95 > 90s)

**Symptoms**:

- Alert: `P95 latency > 90s for 15 minutes`
- Users report slow PDF processing
- Workers show increased `vllm_request_duration`

**Diagnosis**:

```bash
# Check vLLM server metrics
kubectl exec -it vllm-server-xxx -n medical-kg -- curl http://localhost:8000/metrics | grep vllm

# Key metrics:
# - vllm:num_requests_running (should be <20)
# - vllm:num_requests_waiting (queue depth, should be <50)
# - vllm:gpu_cache_usage_perc (should be <90%)

# Check worker metrics
curl -s http://prometheus:9090/api/v1/query?query='mineru_vllm_request_duration_seconds{quantile="0.95"}' | jq
```

**Root Causes & Resolutions**:

**Case 1: vLLM Server Overloaded (Queue Backlog)**

```bash
# Symptom: vllm:num_requests_waiting > 50
# Cause: Too many concurrent workers sending requests

# Solution 1: Scale vLLM server horizontally (if multiple GPUs available)
kubectl scale deployment/vllm-server --replicas=2 -n medical-kg

# Solution 2: Reduce worker concurrency
kubectl set env deployment/mineru-workers CONCURRENCY=2 -n medical-kg  # down from 4

# Solution 3: Increase vLLM batch size (tradeoff: higher latency per request, but more throughput)
kubectl set env deployment/vllm-server MAX_NUM_BATCHED_TOKENS=4096 -n medical-kg
kubectl rollout restart deployment/vllm-server -n medical-kg
```

**Case 2: Large/Complex PDFs**

```bash
# Symptom: Latency spikes correlate with specific PDFs
# Cause: PDFs with many images/tables take longer

# Solution: Identify slow PDFs
kubectl logs -l app=mineru-worker -n medical-kg | grep "processing_time_seconds" | sort -t= -k2 -n | tail -20

# Options:
# - Accept as normal variance (update P99 SLO target)
# - Implement priority queue (small PDFs first)
# - Increase timeout for large PDFs
```

**Case 3: GPU Memory Fragmentation**

```bash
# Symptom: GPU memory usage high (>90%), latency increases over time
# Cause: KV cache fragmentation

# Solution: Restart vLLM server (clears KV cache)
kubectl rollout restart deployment/vllm-server -n medical-kg

# Wait 60s for model reload, latency should improve
```

**Post-Incident**:

- Document findings in incident log
- If recurring: Adjust resource allocation or SLO targets

**Time to Resolution**: 10-30 minutes

---

### Runbook 3: Worker Circuit Breaker Stuck Open

**Symptoms**:

- Alert: `Circuit breaker open for >10 minutes`
- Workers log `CircuitBreakerOpen`, not sending requests to vLLM
- vLLM server healthy but workers not using it

**Diagnosis**:

```bash
# Check circuit breaker state in worker logs
kubectl logs -l app=mineru-worker -n medical-kg | grep circuit_breaker

# Expected in normal operation:
# - circuit_breaker_state=CLOSED (healthy)
# - circuit_breaker_state=HALF_OPEN (testing recovery)

# Problem state:
# - circuit_breaker_state=OPEN (stuck, not recovering)
```

**Resolution**:

```bash
# Cause 1: vLLM server was down, recovered, but circuit breaker recovery timeout not reached
# Solution: Wait for recovery timeout (default: 60s), circuit breaker will auto-transition to HALF_OPEN

# Cause 2: Circuit breaker misconfigured (recovery_timeout too long)
# Solution: Update configuration
kubectl set env deployment/mineru-workers CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30 -n medical-kg
kubectl rollout restart deployment/mineru-workers -n medical-kg

# Cause 3: Bug in circuit breaker logic (rare)
# Solution: Manual workaround - restart workers
kubectl rollout restart deployment/mineru-workers -n medical-kg
```

**Post-Incident**:

- Review circuit breaker configuration
- If bug detected: File issue, develop fix

**Time to Resolution**: 5-10 minutes

---

## Capacity Planning

### Resource Sizing

#### vLLM Server

**Current Sizing** (RTX 5090):

- **GPU**: 1x RTX 5090 (32GB VRAM)
- **CPU**: 8 cores
- **RAM**: 24GB
- **Disk**: 100GB (model weights + cache)

**Capacity**:

- **Peak Throughput**: ~80 PDFs/hour (single server)
- **Concurrent Requests**: 20-30 (limited by GPU memory)
- **Model**: Qwen2.5-VL-7B-Instruct (~14GB VRAM + 10GB KV cache)

**Scaling Triggers**:

- **Horizontal Scale** (add more vLLM servers):
  - Trigger: Queue depth > 50 for >15 minutes
  - Target: Maintain queue depth <20
  - Max replicas: 3 (cost vs benefit)

- **Vertical Scale** (upgrade GPU):
  - Trigger: GPU memory utilization consistently >95%
  - Options: RTX 6000 Ada (48GB), A100 (80GB)

#### MinerU Workers

**Current Sizing**:

- **CPU**: 2 cores per worker
- **RAM**: 4GB per worker
- **Workers**: 8 pods

**Capacity**:

- **Peak Throughput**: ~60 PDFs/hour (with 8 workers @ 7.5 PDFs/hour each)
- **CPU-bound**: PDF parsing, layout analysis

**Scaling Triggers**:

- **Horizontal Scale** (add more workers):
  - Trigger: CPU utilization > 80% for >15 minutes
  - Target: Maintain 60-70% CPU utilization
  - Max workers: 12 (limited by vLLM server capacity)

### Capacity Forecasting

**Growth Assumptions**:

- **Current Volume**: 1,200 PDFs/day (50 PDFs/hour × 24 hours)
- **Growth Rate**: 20% quarterly
- **Peak Usage**: 2x average (80 PDFs/hour during business hours)

**12-Month Forecast**:

| Quarter | Avg PDFs/Day | Peak PDFs/Hour | vLLM Servers | Workers |
|---------|--------------|----------------|--------------|---------|
| Q1 2025 | 1,200 | 80 | 1 | 8 |
| Q2 2025 | 1,440 | 96 | 2 | 12 |
| Q3 2025 | 1,728 | 115 | 2 | 16 |
| Q4 2025 | 2,074 | 138 | 2 | 20 |

**Cost Implications**:

- Q1: $800/month (1 GPU + 8 CPU workers)
- Q2: $1,600/month (2 GPUs + 12 CPU workers)
- Q3: $1,800/month (2 GPUs + 16 CPU workers)
- Q4: $2,000/month (2 GPUs + 20 CPU workers)

**Budget Request**: Plan for 2 GPU nodes by Q2 2025

---

## Maintenance Procedures

### Routine Maintenance Schedule

| Task | Frequency | Duration | Downtime | Owner |
|------|-----------|----------|----------|-------|
| **vLLM Server Restart** (clear cache) | Weekly | 5 min | Yes (5 min) | On-call |
| **Model Update** (Qwen new version) | Quarterly | 30 min | Yes (30 min) | ML Eng |
| **Worker Image Update** (security patches) | Monthly | 15 min | No (rolling) | DevOps |
| **Kubernetes Cluster Upgrade** | Quarterly | 2 hours | No (rolling) | Platform |
| **Certificate Rotation** | Annually | 1 hour | No | Security |

---

### Maintenance Procedure: vLLM Server Upgrade

**Scenario**: Upgrade vLLM from v0.4.0 to v0.5.0

**Preparation** (Day -3):

1. Test new version in staging
2. Review release notes for breaking changes
3. Update deployment manifest with new image tag
4. Schedule maintenance window (announce 48h in advance)

**Execution** (Maintenance Window):

```bash
# 1. Update image tag
kubectl set image deployment/vllm-server vllm=vllm/vllm-openai:v0.5.0 -n medical-kg

# 2. Monitor rollout
kubectl rollout status deployment/vllm-server -n medical-kg

# 3. Validate health
kubectl exec -it vllm-server-xxx -n medical-kg -- curl http://localhost:8000/health

# 4. Test inference
curl -X POST http://vllm-server.medical-kg.svc.cluster.local:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "Test"}]
  }'

# 5. Monitor for 30 minutes
watch -n 30 'kubectl top pod -l app=vllm-server -n medical-kg'

# 6. If issues: Rollback
kubectl rollout undo deployment/vllm-server -n medical-kg
```

**Post-Maintenance**:

- Send completion notification to stakeholders
- Update documentation with new version
- Monitor for 24 hours for regressions

**Estimated Downtime**: 5 minutes (during pod restart)

---

### Maintenance Procedure: Worker Security Patching

**Scenario**: Critical CVE in Python base image, requires immediate patching

**Preparation** (<4 hours):

1. Rebuild worker image with patched base image
2. Push to container registry
3. Test in staging (quick smoke test)

**Execution** (No maintenance window needed):

```bash
# Rolling update (zero downtime)
kubectl set image deployment/mineru-workers mineru-worker=ghcr.io/your-org/mineru-worker:v1.2.3-patched -n medical-kg

# Monitor rolling update
kubectl rollout status deployment/mineru-workers -n medical-kg

# Validate all pods updated
kubectl get pods -l app=mineru-worker -n medical-kg -o jsonpath='{.items[*].spec.containers[*].image}'
```

**Post-Patching**:

- Re-scan image to confirm CVE remediated
- Update tracking spreadsheet

**Estimated Downtime**: 0 (rolling update)

---

## Backup and Disaster Recovery

### Backup Strategy

#### What to Backup

1. **vLLM Server Configuration**:
   - Kubernetes manifests (Deployment, Service, ConfigMap, Secret)
   - Stored in Git repository (Infrastructure as Code)

2. **Model Weights**:
   - Qwen2.5-VL-7B-Instruct weights cached in PersistentVolume
   - Backup: Snapshot PersistentVolume weekly
   - Retention: 4 weeks

3. **Worker Configuration**:
   - Kubernetes manifests
   - Stored in Git repository

4. **Processed PDFs** (Output Data):
   - Stored in MinIO (object storage)
   - MinIO handles replication (3x redundancy)
   - No additional backup needed

#### Backup Schedule

| Asset | Frequency | Retention | Tool |
|-------|-----------|-----------|------|
| **K8s Manifests** | On change | Forever (Git) | Git |
| **Model Weights PV** | Weekly | 4 weeks | Velero |
| **Prometheus Metrics** | Daily | 90 days | Thanos |

---

### Disaster Recovery Scenarios

#### DR Scenario 1: vLLM Server Pod Deleted

**RTO** (Recovery Time Objective): 5 minutes
**RPO** (Recovery Point Objective): 0 (no data loss)

**Recovery Procedure**:

```bash
# Kubernetes automatically recreates pod
# No action needed (self-healing)

# Validate recovery
kubectl get pods -l app=vllm-server -n medical-kg
# Wait for STATUS: Running, READY: 1/1
```

---

#### DR Scenario 2: GPU Node Failure

**RTO**: 10 minutes
**RPO**: 0

**Recovery Procedure**:

```bash
# 1. Kubernetes detects node failure, cordons node
# 2. vLLM pod rescheduled to another GPU node (if available)

# If no other GPU node available:
# 3. Provision new GPU node (GKE/EKS auto-scaling or manual)
# 4. vLLM pod schedules to new node
# 5. Wait for model loading (60s)

# Validate
kubectl get pods -l app=vllm-server -n medical-kg -o wide
```

**Mitigation**: Maintain 2 GPU nodes for redundancy (cost tradeoff)

---

#### DR Scenario 3: Model Weights PersistentVolume Corrupted

**RTO**: 30 minutes
**RPO**: 7 days (weekly backup)

**Recovery Procedure**:

```bash
# 1. Restore PV from Velero backup
velero restore create --from-backup vllm-model-weights-backup-<date>

# 2. Restart vLLM server to mount restored PV
kubectl rollout restart deployment/vllm-server -n medical-kg

# 3. Validate model loading
kubectl logs -f vllm-server-xxx -n medical-kg | grep "Successfully loaded model"

# Alternative: Re-download model from HuggingFace (slower, 15 min)
kubectl delete pvc vllm-model-cache -n medical-kg
kubectl apply -f ops/k8s/base/pvc-vllm-model-cache.yaml
kubectl rollout restart deployment/vllm-server -n medical-kg
```

---

#### DR Scenario 4: Entire Kubernetes Cluster Failure

**RTO**: 4 hours
**RPO**: 24 hours

**Recovery Procedure**:

1. **Provision new cluster** (1 hour): Use Infrastructure-as-Code (Terraform)
2. **Restore configuration** (30 min): Apply Kubernetes manifests from Git
3. **Restore data** (2 hours): Restore PVs from backups, sync MinIO from replicas
4. **Validate** (30 min): Smoke test, process 10 PDFs end-to-end

**Prevention**: Multi-region deployment (future)

---

## On-Call Procedures

### On-Call Schedule

- **Rotation**: Weekly rotation among 5 engineers
- **Hours**: 24/7 coverage
- **Escalation**: If on-call engineer doesn't respond in 15 minutes, escalate to Engineering Lead

### On-Call Responsibilities

1. **Respond to Alerts**: Within 15 minutes for Sev2, immediately for Sev1
2. **Incident Management**: Follow runbooks, document actions
3. **Communication**: Update status in `#medical-kg-incidents`
4. **Escalation**: Engage specialists (ML Eng, Security) as needed
5. **Post-Incident**: Write incident summary within 24 hours

### On-Call Handoff Checklist

**Outgoing On-Call**:

- [ ] Review open incidents (if any)
- [ ] Share runbook updates or learnings
- [ ] Verify incoming on-call has access (Grafana, Kubernetes, PagerDuty)

**Incoming On-Call**:

- [ ] Test PagerDuty alert delivery (send test page)
- [ ] Review SLO dashboard, note current state
- [ ] Read incident log from previous week
- [ ] Review upcoming maintenance schedule

---

## Performance Tuning

### Tuning vLLM Server

#### Batch Size Tuning

**Goal**: Maximize throughput without exceeding GPU memory

**Procedure**:

```bash
# Experiment with max_num_batched_tokens
# Lower = less latency, higher = more throughput

# Current setting: 4096 tokens
kubectl set env deployment/vllm-server MAX_NUM_BATCHED_TOKENS=4096 -n medical-kg

# Experiment: Increase to 8192
kubectl set env deployment/vllm-server MAX_NUM_BATCHED_TOKENS=8192 -n medical-kg
kubectl rollout restart deployment/vllm-server -n medical-kg

# Monitor: GPU memory usage, throughput, latency
watch -n 10 'kubectl exec -it vllm-server-xxx -n medical-kg -- nvidia-smi'

# If GPU OOM: Decrease batch size
# If GPU memory <80%: Can increase further
```

**Recommended Values**:

- RTX 5090 (32GB): 4096-8192 tokens
- A100 (80GB): 16384-32768 tokens

#### GPU Memory Utilization

**Goal**: Maximize KV cache size without risking OOM

**Current Setting**: 0.92 (92% of GPU memory)

**Tuning**:

```bash
# If frequent OOMs: Reduce to 0.85
kubectl set env deployment/vllm-server GPU_MEMORY_UTILIZATION=0.85 -n medical-kg

# If stable and underutilized: Increase to 0.95 (aggressive)
kubectl set env deployment/vllm-server GPU_MEMORY_UTILIZATION=0.95 -n medical-kg
```

### Tuning Workers

#### Worker Concurrency

**Goal**: Balance worker throughput vs vLLM server load

**Procedure**:

```bash
# Increase concurrency (more PDFs processed in parallel per worker)
kubectl set env deployment/mineru-workers WORKER_CONCURRENCY=4 -n medical-kg

# Monitor: vLLM queue depth, latency
# If vLLM queue depth consistently >50: Reduce worker concurrency
# If vLLM queue depth <10: Can increase worker concurrency
```

**Recommended**: 2-4 concurrent PDFs per worker (8 workers = 16-32 concurrent requests to vLLM)

---

**Document Control**:

- **Version**: 1.0
- **Last Updated**: 2025-10-08
- **Next Review**: After deployment + quarterly
- **Owner**: Operations Lead
- **Approvers**: Engineering Lead, SRE Lead
