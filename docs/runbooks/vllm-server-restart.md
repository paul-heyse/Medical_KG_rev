# Runbook: Restarting the vLLM Server

This runbook explains how to safely restart the vLLM server that serves MinerU workers.

## Preconditions

- MinerU jobs draining or paused
- Operator access to the `medical-kg` namespace
- Grafana + Prometheus available for monitoring

## Steps

1. **Scale MinerU workers to zero**
   ```bash
   kubectl scale deployment/mineru-workers --replicas=0 -n medical-kg
   kubectl rollout status deployment/mineru-workers -n medical-kg
   ```

2. **Restart the vLLM deployment**
   ```bash
   kubectl rollout restart deployment/vllm-server -n medical-kg
   kubectl rollout status deployment/vllm-server -n medical-kg
   ```

3. **Verify health**
   ```bash
   kubectl exec -n medical-kg deployment/vllm-server -- curl -sf http://localhost:8000/health
   ```

4. **Scale MinerU workers back up**
   ```bash
   kubectl scale deployment/mineru-workers --replicas=8 -n medical-kg
   kubectl rollout status deployment/mineru-workers -n medical-kg
   ```

5. **Monitor metrics**
   - Grafana dashboard: `MinerU / vLLM`
   - Prometheus queries:
     - `mineru_vllm_request_duration_seconds:histogram_quantile(0.95, sum(rate(...)[5m]))`
     - `mineru_vllm_circuit_breaker_state`

6. **Resume job submission** once metrics stabilise and worker readiness probes are healthy.

## Rollback

If the restart fails, roll back the vLLM deployment:

```bash
kubectl rollout undo deployment/vllm-server -n medical-kg
```

Then scale workers back to the previous replica count.

## Verification Checklist

- [ ] vLLM `/health` returns HTTP 200
- [ ] MinerU workers `READY`
- [ ] Circuit breaker metric returns to `0` (closed)
- [ ] No `mineru_vllm_client_failures_total` increase during restart window
