# Security Implementation: MinerU vLLM Split-Container Architecture

**Change ID**: `update-mineru-vllm-split-container`
**Version**: 1.0
**Date**: 2025-10-08
**Classification**: Confidential - Security Architecture

---

## Table of Contents

1. [Threat Model](#threat-model)
2. [Security Controls](#security-controls)
3. [Network Security](#network-security)
4. [Authentication & Authorization](#authentication--authorization)
5. [Secrets Management](#secrets-management)
6. [Security Testing](#security-testing)
7. [Incident Response](#incident-response)
8. [Compliance](#compliance)

---

## Threat Model

### STRIDE Analysis

#### Spoofing

**Threat**: Unauthorized pod impersonates MinerU worker to access vLLM server

**Impact**: Unauthorized access to VLM inference, potential data exfiltration

**Mitigation**:

- NetworkPolicy restricts vLLM access to MinerU worker pods only (label-based)
- Optional: API key authentication on vLLM server
- Pod Identity: Service Account with minimal RBAC permissions

**Residual Risk**: Low (network-level isolation effective)

#### Tampering

**Threat**: Attacker modifies vLLM HTTP requests/responses in transit

**Impact**: Corrupted inference results, potential injection attacks

**Mitigation**:

- mTLS for pod-to-pod communication (Istio service mesh)
- OR HTTP over internal network (trusted cluster network)
- Input validation on worker side before sending to vLLM

**Residual Risk**: Very Low (internal cluster network trusted)

#### Repudiation

**Threat**: Malicious actor denies making vLLM requests

**Impact**: Audit trail gaps, difficulty attributing actions

**Mitigation**:

- Structured logging with correlation_id, tenant_id, user_id for all vLLM calls
- Centralized log aggregation (Loki, CloudWatch)
- Log retention policy: 90 days minimum

**Residual Risk**: Very Low (comprehensive logging)

#### Information Disclosure

**Threat 1**: vLLM model weights exposed via API
**Impact**: Model theft, competitive disadvantage

**Mitigation**:

- vLLM server not exposed outside cluster (no Ingress)
- NetworkPolicy blocks all non-worker traffic
- Model weights stored in read-only PersistentVolume

**Threat 2**: Sensitive data in logs (PII, PHI)
**Impact**: HIPAA violation, data breach

**Mitigation**:

- Structured logging with PII redaction filters
- Log scrubbing pipeline (regex-based redaction)
- Audit log review: weekly automated scan for leaked secrets

**Residual Risk**: Low-Medium (requires strict log hygiene)

#### Denial of Service

**Threat 1**: Worker flood attack on vLLM server
**Impact**: vLLM server overwhelmed, service unavailable

**Mitigation**:

- Worker connection pool limits (10 max connections per worker)
- vLLM server request queue limits (max 100 pending requests)
- Kubernetes resource limits on workers (prevent accidental DoS)

**Threat 2**: Large/malicious PDF crashes vLLM
**Impact**: vLLM pod OOM, service disruption

**Mitigation**:

- PDF size limits (50MB max, enforced at ingestion)
- vLLM GPU memory limits (24GB hard limit)
- PodDisruptionBudget ensures 1 vLLM pod always running (if 2+ replicas)

**Residual Risk**: Medium (malicious PDFs still possible)

#### Elevation of Privilege

**Threat**: Container breakout via GPU driver vulnerability
**Impact**: Host compromise, cluster-wide breach

**Mitigation**:

- Non-root containers (user 1000:1000)
- Read-only root filesystem where possible
- AppArmor/Seccomp profiles
- Regular NVIDIA driver updates
- Container runtime security (gVisor if extreme isolation needed)

**Residual Risk**: Low (defense-in-depth applied)

---

## Security Controls

### Container Security

#### Base Image Hardening

**vLLM Server**:

```dockerfile
# Use official vLLM image (regularly updated by vLLM project)
FROM vllm/vllm-openai:latest

# Run as non-root user
USER 1000:1000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

**MinerU Worker**:

```dockerfile
# Use minimal Python base
FROM python:3.12-slim

# Create non-root user
RUN groupadd -r mineru --gid=1000 && \
    useradd -r -g mineru --uid=1000 --home-dir=/app mineru

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY --chown=mineru:mineru src/ /app/

# Run as non-root
USER 1000:1000
WORKDIR /app

CMD ["python", "-m", "Medical_KG_rev.services.mineru.worker"]
```

#### Kubernetes Security Context

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: vllm
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false  # vLLM needs /tmp writes
          capabilities:
            drop: ["ALL"]
            add: []  # No capabilities needed

        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 24Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
```

#### Vulnerability Scanning

**Trivy Scan (CI/CD)**:

```bash
# Scan vLLM server image
trivy image --severity HIGH,CRITICAL vllm/vllm-openai:latest

# Scan MinerU worker image
trivy image --severity HIGH,CRITICAL ghcr.io/your-org/mineru-worker:latest

# Fail build if HIGH/CRITICAL vulnerabilities found
trivy image --exit-code 1 --severity CRITICAL ghcr.io/your-org/mineru-worker:latest
```

**Scan Schedule**:

- **CI/CD**: Every build
- **Production**: Daily scan of running images
- **Alert**: Security team notified if new CVE detected

---

## Network Security

### Kubernetes NetworkPolicy

#### Policy 1: Isolate vLLM Server

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vllm-server-isolation
  namespace: medical-kg
spec:
  podSelector:
    matchLabels:
      app: vllm-server
  policyTypes:
  - Ingress
  - Egress

  ingress:
  # Allow ONLY from MinerU workers
  - from:
    - podSelector:
        matchLabels:
          app: mineru-worker
    ports:
    - protocol: TCP
      port: 8000

  # Allow from Prometheus (metrics scraping)
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
      podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 8000  # /metrics endpoint

  egress:
  # Allow outbound to HuggingFace (model download)
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 443

  # Allow DNS resolution
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
```

#### Policy 2: MinerU Worker Restrictions

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mineru-worker-restrictions
  namespace: medical-kg
spec:
  podSelector:
    matchLabels:
      app: mineru-worker
  policyTypes:
  - Egress

  egress:
  # Allow to vLLM server
  - to:
    - podSelector:
        matchLabels:
          app: vllm-server
    ports:
    - protocol: TCP
      port: 8000

  # Allow to Kafka
  - to:
    - podSelector:
        matchLabels:
          app: kafka
    ports:
    - protocol: TCP
      port: 9092

  # Allow to MinIO (object storage)
  - to:
    - podSelector:
        matchLabels:
          app: minio
    ports:
    - protocol: TCP
      port: 9000

  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
```

#### Policy Testing

```bash
# Test: Worker can access vLLM server
kubectl exec -it mineru-worker-xxx -n medical-kg -- curl http://vllm-server:8000/health
# Expected: 200 OK

# Test: External pod CANNOT access vLLM server
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- curl http://vllm-server.medical-kg.svc.cluster.local:8000/health
# Expected: Timeout (connection blocked by NetworkPolicy)

# Test: Worker CANNOT access internet (except DNS)
kubectl exec -it mineru-worker-xxx -n medical-kg -- curl https://google.com
# Expected: Timeout (blocked by NetworkPolicy)
```

### Service Mesh (Optional: Istio)

If using Istio for mTLS:

```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: medical-kg
spec:
  mtls:
    mode: STRICT  # Enforce mTLS for all pod-to-pod

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: vllm-server-authz
  namespace: medical-kg
spec:
  selector:
    matchLabels:
      app: vllm-server
  action: ALLOW
  rules:
  - from:
    - source:
        principals:
        - "cluster.local/ns/medical-kg/sa/mineru-worker"  # Service Account
```

---

## Authentication & Authorization

### vLLM Server API Key (Optional)

**When to Enable**:

- Multi-tenant vLLM server (multiple teams sharing)
- Zero-trust network model

**Implementation**:

1. **Generate API Key**:

   ```bash
   # Generate random API key
   API_KEY=$(openssl rand -base64 32)

   # Store in Kubernetes secret
   kubectl create secret generic vllm-api-key \
     --from-literal=api-key=$API_KEY \
     -n medical-kg
   ```

2. **Configure vLLM Server**:

   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: vllm-server
   spec:
     template:
       spec:
         containers:
         - name: vllm
           env:
           - name: VLLM_API_KEY
             valueFrom:
               secretKeyRef:
                 name: vllm-api-key
                 key: api-key

           command:
           - python
           - -m
           - vllm.entrypoints.openai.api_server
           - --api-key
           - $(VLLM_API_KEY)
   ```

3. **Configure MinerU Workers**:

   ```python
   # src/Medical_KG_rev/services/mineru/vllm_client.py

   class VLLMClient:
       def __init__(self, base_url: str, api_key: str | None = None):
           self.base_url = base_url
           self.api_key = api_key or os.getenv("VLLM_API_KEY")

           self.client = httpx.AsyncClient(
               base_url=base_url,
               headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
           )
   ```

**Recommendation**: Enable API key auth in production for defense-in-depth, even with NetworkPolicy.

### RBAC Permissions

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mineru-worker
  namespace: medical-kg

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mineru-worker-role
  namespace: medical-kg
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]  # Only read pods (for health checks)
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get"]  # Read config

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mineru-worker-binding
  namespace: medical-kg
subjects:
- kind: ServiceAccount
  name: mineru-worker
roleRef:
  kind: Role
  name: mineru-worker-role
  apiGroup: rbac.authorization.k8s.io
```

---

## Secrets Management

### Strategy: Kubernetes Secrets + External Secrets Operator

**For Sensitive Values**:

- vLLM API key
- HuggingFace token (model download)
- MinIO credentials
- Kafka credentials

#### Option 1: Kubernetes Native Secrets

```bash
# Create secret
kubectl create secret generic vllm-secrets \
  --from-literal=api-key=$VLLM_API_KEY \
  --from-literal=hf-token=$HF_TOKEN \
  -n medical-kg

# Reference in Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  template:
    spec:
      containers:
      - name: vllm
        env:
        - name: VLLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: vllm-secrets
              key: api-key
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: vllm-secrets
              key: hf-token
```

#### Option 2: HashiCorp Vault + External Secrets Operator

**For Production (Recommended)**:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: medical-kg
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "medical-kg-role"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: vllm-secrets
  namespace: medical-kg
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: vllm-secrets
    creationPolicy: Owner
  data:
  - secretKey: api-key
    remoteRef:
      key: medical-kg/vllm
      property: api-key
  - secretKey: hf-token
    remoteRef:
      key: medical-kg/vllm
      property: hf-token
```

### Secrets Rotation

**Rotation Schedule**:

- vLLM API key: Every 90 days
- HuggingFace token: Every 180 days

**Rotation Procedure**:

1. Generate new secret in Vault
2. Update Kubernetes secret (External Secrets Operator auto-syncs)
3. Rolling restart of vLLM server and workers
4. Validate new secret works
5. Revoke old secret after 24 hours

---

## Security Testing

### Penetration Testing Checklist

#### Network Isolation Testing

- [ ] **Test 1**: Attempt to access vLLM from external pod → Expected: Blocked
- [ ] **Test 2**: Attempt to access vLLM from worker pod → Expected: Allowed
- [ ] **Test 3**: Port scan vLLM service from outside cluster → Expected: No open ports

#### Authentication Testing

- [ ] **Test 4**: Send vLLM request without API key → Expected: 401 Unauthorized
- [ ] **Test 5**: Send vLLM request with invalid API key → Expected: 403 Forbidden
- [ ] **Test 6**: Send vLLM request with valid API key → Expected: 200 OK

#### Input Validation Testing

- [ ] **Test 7**: Send malformed JSON to vLLM → Expected: 400 Bad Request
- [ ] **Test 8**: Send extremely large request (>100MB) → Expected: 413 Payload Too Large
- [ ] **Test 9**: Send SQL injection in prompt → Expected: Sanitized, no code execution

#### Container Escape Testing

- [ ] **Test 10**: Attempt to write to read-only filesystem → Expected: Permission denied
- [ ] **Test 11**: Attempt to escalate privileges → Expected: Blocked by seccomp
- [ ] **Test 12**: Attempt to access host filesystem → Expected: Blocked

### Security Scanning

**OWASP ZAP (API Scanning)**:

```bash
# Scan vLLM server API (in staging)
docker run -t owasp/zap2docker-stable zap-api-scan.py \
  -t http://vllm-server.medical-kg-staging.svc.cluster.local:8000/openapi.json \
  -f openapi
```

**Kube-bench (CIS Kubernetes Benchmark)**:

```bash
# Run kube-bench on cluster
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml

# Review results
kubectl logs -l app=kube-bench
```

---

## Incident Response

### Security Incident Classification

| Severity | Examples | Response Time | Escalation |
|----------|----------|---------------|------------|
| **Sev1 (Critical)** | Data breach, unauthorized access | Immediate | CEO, Legal, Security Lead |
| **Sev2 (High)** | Suspicious activity, failed pentest | <1 hour | Security Lead, Engineering Lead |
| **Sev3 (Medium)** | Security scan findings (High CVE) | <4 hours | Security team |
| **Sev4 (Low)** | Policy violations, Medium CVE | <1 day | Security team |

### Incident Response Playbook

#### Scenario: Unauthorized vLLM Access Detected

**Detection**: Logs show vLLM requests from unknown pod

**Response Steps**:

1. **Isolate** (5 min):

   ```bash
   # Block all traffic to vLLM server
   kubectl apply -f - <<EOF
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: vllm-server-emergency-block
     namespace: medical-kg
   spec:
     podSelector:
       matchLabels:
         app: vllm-server
     policyTypes:
     - Ingress
     ingress: []  # Block all ingress
   EOF
   ```

2. **Investigate** (15 min):
   - Review logs: Identify unauthorized pod
   - Check Kubernetes audit logs: Who created unauthorized pod?
   - Analyze traffic: What data was accessed?

3. **Contain** (10 min):

   ```bash
   # Delete unauthorized pod
   kubectl delete pod <unauthorized-pod> -n medical-kg

   # Rotate vLLM API key (if enabled)
   kubectl create secret generic vllm-api-key --from-literal=api-key=$(openssl rand -base64 32) --dry-run=client -o yaml | kubectl apply -f -

   # Rolling restart vLLM + workers
   kubectl rollout restart deployment/vllm-server deployment/mineru-workers -n medical-kg
   ```

4. **Recover** (20 min):
   - Restore normal NetworkPolicy
   - Validate system functionality
   - Monitor for 1 hour

5. **Post-Incident** (24 hours):
   - Root cause analysis
   - Incident report
   - Remediation plan
   - Update runbooks

---

## Compliance

### HIPAA Compliance (if applicable)

**Relevant Requirements**:

- **164.308(a)(1)**: Security management process
- **164.312(a)(1)**: Access control (NetworkPolicy, RBAC)
- **164.312(d)**: Person/entity authentication (API keys)
- **164.312(e)(1)**: Transmission security (mTLS optional)

**Controls Implemented**:

- ✅ Access control: NetworkPolicy + RBAC
- ✅ Audit logs: Structured logging with correlation IDs
- ✅ Encryption in transit: Kubernetes internal network (trusted)
- ✅ Encryption at rest: Persistent volumes encrypted (cloud provider)

**Audit Checklist**:

- [ ] NetworkPolicies enforce least privilege
- [ ] All vLLM API calls logged with tenant_id
- [ ] Logs retained for 90 days minimum
- [ ] Security scan results reviewed monthly
- [ ] Penetration test conducted annually

### SOC 2 Compliance (if applicable)

**CC6.1 - Logical and Physical Access Controls**:

- NetworkPolicy enforcement
- RBAC least privilege
- API key authentication

**CC7.2 - System Monitoring**:

- Prometheus metrics
- Centralized logging
- Alerting on anomalies

---

## Security Checklist (Pre-Deployment)

- [ ] Vulnerability scans completed (no HIGH/CRITICAL)
- [ ] NetworkPolicies deployed and tested
- [ ] Secrets stored in Vault (or Kubernetes Secrets encrypted at rest)
- [ ] Non-root containers enforced
- [ ] Read-only root filesystem where possible
- [ ] Penetration test completed
- [ ] Security incident response runbook reviewed
- [ ] Compliance requirements documented
- [ ] Security team sign-off obtained

---

**Document Control**:

- **Version**: 1.0
- **Classification**: Confidential
- **Last Updated**: 2025-10-08
- **Next Review**: After deployment + quarterly
- **Owner**: Security Lead
- **Approvers**: CISO, Engineering Lead
