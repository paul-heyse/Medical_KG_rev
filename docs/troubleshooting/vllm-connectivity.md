# Troubleshooting Guide: MinerU ↔ vLLM Connectivity

> **Legacy Notice:** MinerU has been decommissioned in favour of the Docling Gemma3 VLM pipeline. This guide is retained solely
> for emergency rollback scenarios triggered via `scripts/rollback_to_mineru.sh`.

This guide catalogues common failure scenarios when MinerU workers communicate with the
vLLM server.

## 1. Worker Startup Failure

**Symptoms**
- Worker pod fails readiness probe
- Logs show `vLLM server health check failed on startup`

**Diagnostics**
- `kubectl logs deployment/mineru-workers -n medical-kg --tail=100`
- `kubectl get pods -n medical-kg -l app=vllm-server`
- `kubectl exec -n medical-kg deployment/mineru-workers -- curl -sf http://vllm-server:8000/health`

**Resolution**
- Ensure the vLLM pod is running and ready
- Check `networkpolicy-vllm-server.yaml` to confirm worker namespace selector
- Verify DNS resolution inside the worker pod (`nslookup vllm-server`)

## 2. Elevated Latency / Timeouts

**Symptoms**
- `mineru_vllm_client_failures_total{error_type="timeout"}` increasing
- Circuit breaker transitions to `OPEN`

**Diagnostics**
- Grafana: inspect the `vLLM Queue Depth` panel
- Prometheus query: `vllm_time_to_first_token_seconds` and `gpu_utilization`
- Review worker logs for `mineru.vllm.retry` warnings

**Resolution**
- Scale the vLLM deployment vertically (larger GPU) or horizontally (tensor parallelism)
- Increase `connection_pool_size` and `retry_attempts` in `config/mineru.yaml`
- Confirm no other workloads are saturating the GPU node

## 3. Authentication / TLS Issues

**Symptoms**
- Workers report `HTTP 403` or `HTTP 495`

**Diagnostics**
- Inspect vLLM server logs for mTLS or token errors
- Confirm configuration in `ops/k8s/base/configmap-vllm-server.yaml`
- Check Istio/NGINX ingress annotations if traffic passes through a mesh

**Resolution**
- Regenerate and distribute TLS certificates if mutual TLS enabled
- Ensure worker environment includes updated auth headers via ConfigMap or Secret

## 4. Model Download Failures

**Symptoms**
- vLLM pod stuck pulling model
- Logs show `Read timeout` from Hugging Face

**Diagnostics**
- `kubectl logs deployment/vllm-server -n medical-kg`
- Check PVC mount `pvc-huggingface-cache`
- Inspect egress firewall rules

**Resolution**
- Pre-seed the model artefacts into the PVC
- Allow outbound HTTPS to `huggingface.co`
- Increase `connection_timeout_seconds` in `config/mineru.yaml`

## 5. Persistent Circuit Breaker Open State

**Symptoms**
- `mineru_vllm_circuit_breaker_state` remains at `2`
- Workers immediately fail with `Circuit breaker is open`

**Diagnostics**
- Inspect worker logs for the first failure event
- Query Prometheus for `mineru_vllm_client_failures_total` to identify root cause

**Resolution**
- Validate network connectivity and DNS
- If vLLM server recovered, restart MinerU workers to reset the breaker state
- Consider increasing `recovery_timeout_seconds` after verifying stability

Document root causes and mitigations in the incident tracker to aid future responses.
