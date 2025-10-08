# Risk Register and Operational Readiness: MinerU vLLM Split-Container

**Change ID**: `update-mineru-vllm-split-container`
**Version**: 1.0
**Date**: 2025-10-08

---

## Risk Register

### Risk Assessment Matrix

**Probability Scale**:

- **High** (H): >60% likelihood
- **Medium** (M): 30-60% likelihood
- **Low** (L): <30% likelihood

**Impact Scale**:

- **High** (H): Service outage, data loss, security breach
- **Medium** (M): Performance degradation, delayed timeline
- **Low** (L): Minor inconvenience, easily mitigated

**Risk Score**: Probability × Impact

| P \ I | High (3) | Medium (2) | Low (1) |
|-------|----------|------------|---------|
| **High (3)** | 9 (Critical) | 6 (High) | 3 (Medium) |
| **Medium (2)** | 6 (High) | 4 (Medium) | 2 (Low) |
| **Low (1)** | 3 (Medium) | 2 (Low) | 1 (Low) |

---

### Risk 1: vLLM Server Single Point of Failure

**Risk ID**: RISK-001
**Category**: Technical
**Description**: Centralized vLLM server becomes single point of failure. If server crashes, all PDF processing stops.

**Probability**: Medium (30%)
**Impact**: High (Service outage)
**Risk Score**: 6 (High)

**Mitigation**:

1. **Health Checks**: Kubernetes liveness/readiness probes auto-restart failed pods
2. **Circuit Breakers**: Workers gracefully degrade when vLLM unavailable
3. **Fast Recovery**: vLLM pod restarts in <5 minutes (model preloaded in PV)
4. **Monitoring**: Alert on vLLM server down within 2 minutes

**Contingency Plan**:

- Deploy 2 vLLM replicas with load balancing (if budget allows)
- Keep monolithic fallback deployment available for 30 days post-migration

**Owner**: Engineering Lead
**Review Date**: Weekly during rollout, monthly post-deployment

---

### Risk 2: Quality Regression in PDF Processing

**Risk ID**: RISK-002
**Category**: Quality
**Description**: HTTP communication introduces latency/errors that degrade PDF extraction quality.

**Probability**: Low (15%)
**Impact**: High (User-facing quality issues)
**Risk Score**: 3 (Medium)

**Mitigation**:

1. **Quality Baseline**: Establish baseline with 100 PDFs before migration
2. **Automated Comparison**: Compare outputs with baseline, alert if <95% match
3. **Phased Rollout**: 10% → 50% → 100% allows early detection
4. **Rollback Plan**: Immediate rollback if quality drops >5%

**Contingency Plan**:

- If quality regression detected: Rollback to monolithic within 15 minutes
- Investigate root cause (HTTP serialization, timeout issues)
- Fix in staging, re-validate with 200 PDFs before retry

**Owner**: QA Lead
**Review Date**: After each rollout phase

---

### Risk 3: vLLM Server GPU Out-of-Memory (OOM)

**Risk ID**: RISK-003
**Category**: Technical
**Description**: Under heavy load, vLLM server GPU memory exhausted, causing crashes.

**Probability**: Medium (40%)
**Impact**: Medium (Temporary service disruption)
**Risk Score**: 4 (Medium)

**Mitigation**:

1. **Conservative GPU Memory**: Set `gpu_memory_utilization=0.92` (not 0.95)
2. **Request Limits**: Cap concurrent requests at 30
3. **Monitoring**: Alert if GPU memory >95% for 5 minutes
4. **Auto-Restart**: Kubernetes restarts pod on OOM (self-healing)

**Contingency Plan**:

- If recurring OOMs: Reduce batch size or upgrade GPU (RTX 5090 → A100)
- Short-term: Scale workers down to reduce request rate

**Owner**: SRE Lead
**Review Date**: Weekly for first month, then monthly

---

### Risk 4: Network Latency Spikes Between Workers and vLLM

**Risk ID**: RISK-004
**Category**: Performance
**Description**: Network congestion or misconfiguration causes high latency in HTTP requests.

**Probability**: Low (20%)
**Impact**: Medium (Performance degradation)
**Risk Score**: 2 (Low)

**Mitigation**:

1. **Local Network**: vLLM and workers in same Kubernetes cluster (low latency)
2. **Connection Pooling**: HTTP keepalive connections (no reconnect overhead)
3. **Retries**: Exponential backoff handles transient network issues
4. **Monitoring**: Track P95 HTTP request latency

**Contingency Plan**:

- If latency spikes detected: Check network policies, Kubernetes CNI health
- Temporary: Increase timeout thresholds
- Long-term: Investigate Istio service mesh for traffic management

**Owner**: Platform Team
**Review Date**: Monthly

---

### Risk 5: Implementation Timeline Overrun

**Risk ID**: RISK-005
**Category**: Project Management
**Description**: Implementation takes longer than planned (5 weeks), delaying deployment.

**Probability**: Medium (50%)
**Impact**: Low (Delayed benefits, no service impact)
**Risk Score**: 2 (Low)

**Mitigation**:

1. **Detailed Tasks**: 216 tasks with time estimates in tasks.md
2. **Parallel Work**: Multiple engineers work on independent phases
3. **Weekly Reviews**: Track progress, identify blockers early
4. **Buffer**: Built-in 1-week buffer in timeline

**Contingency Plan**:

- If critical path delayed: Descope non-essential tasks (e.g., advanced monitoring)
- If >2 weeks delay: Re-evaluate go/no-go decision

**Owner**: Project Manager
**Review Date**: Weekly stand-up

---

### Risk 6: Security Vulnerability in vLLM Server

**Risk ID**: RISK-006
**Category**: Security
**Description**: vLLM server exposes unintended attack surface (model theft, unauthorized access).

**Probability**: Low (10%)
**Impact**: High (Security breach, compliance violation)
**Risk Score**: 3 (Medium)

**Mitigation**:

1. **Network Isolation**: Kubernetes NetworkPolicy restricts access to workers only
2. **No External Exposure**: vLLM not exposed via Ingress
3. **API Key Auth**: Optional API key authentication enabled
4. **Penetration Testing**: Security team tests before production

**Contingency Plan**:

- If vulnerability detected: Immediate patch or disable vLLM, revert to monolithic
- Security incident response per SECURITY.md

**Owner**: Security Lead
**Review Date**: Pre-deployment + quarterly

---

### Risk 7: Worker Circuit Breaker Mistuned

**Risk ID**: RISK-007
**Category**: Technical
**Description**: Circuit breaker too sensitive (false opens) or too lenient (doesn't open when needed).

**Probability**: Medium (30%)
**Impact**: Medium (False alarms or delayed failure detection)
**Risk Score**: 4 (Medium)

**Mitigation**:

1. **Tuning Testing**: Test circuit breaker in staging with chaos experiments
2. **Metrics**: Monitor circuit breaker state changes, false positive rate
3. **Adjustable Config**: Can tune thresholds via ConfigMap without code changes

**Contingency Plan**:

- If too sensitive: Increase failure threshold (5 → 10)
- If too lenient: Decrease recovery timeout (60s → 30s)
- Iterate based on production data

**Owner**: Engineering Lead
**Review Date**: After Phase 1 (10% rollout)

---

### Risk 8: Cost Overrun (Infrastructure)

**Risk ID**: RISK-008
**Category**: Financial
**Description**: GPU costs higher than estimated, budget exceeded.

**Probability**: Medium (40%)
**Impact**: Low (Budget variance, not service impact)
**Risk Score**: 2 (Low)

**Mitigation**:

1. **Cost Estimate Validation**: Confirm pricing with cloud provider
2. **Committed Use Discounts**: 1-year commitment for 25% discount
3. **Monitoring**: Track actual costs weekly, alert if >10% over budget

**Contingency Plan**:

- If costs exceed: Optimize (preemptible VMs, right-sizing)
- Request budget variance approval
- Worst case: Revert to monolithic if cost unjustifiable

**Owner**: Finance Manager
**Review Date**: Monthly

---

### Risk 9: Team Knowledge Gap

**Risk ID**: RISK-009
**Category**: Operational
**Description**: Operations team unfamiliar with vLLM server troubleshooting, slow incident response.

**Probability**: Medium (40%)
**Impact**: Medium (Increased MTTR)
**Risk Score**: 4 (Medium)

**Mitigation**:

1. **Training**: 2-hour hands-on workshop for ops team (see TRAINING_AND_COMMUNICATION.md)
2. **Runbooks**: Detailed runbooks in OPERATIONS.md
3. **Shadowing**: Ops engineers shadow implementation team
4. **On-Call Support**: Engineering on-call for first 2 weeks

**Contingency Plan**:

- If knowledge gap persists: Additional training sessions
- Extend engineering on-call support to 4 weeks
- Pair ops engineers with engineers during incidents

**Owner**: Operations Lead
**Review Date**: After training, then monthly

---

### Risk 10: Rollback Complexity

**Risk ID**: RISK-010
**Category**: Operational
**Description**: Rollback procedure fails or takes too long, extending outage.

**Probability**: Low (15%)
**Impact**: High (Extended downtime)
**Risk Score**: 3 (Medium)

**Mitigation**:

1. **Rollback Testing**: Test rollback procedure in staging before production
2. **Feature Flag**: Fast rollback via config change (5 minutes)
3. **Documented Procedure**: Step-by-step in DEPLOYMENT_STRATEGY.md
4. **Dry Run**: Practice rollback drill with ops team

**Contingency Plan**:

- If rollback fails: Escalate to Engineering Lead immediately
- Nuclear option: Delete vLLM deployment, scale old monolithic deployment

**Owner**: Deployment Lead
**Review Date**: Pre-deployment dry run

---

## Risk Summary Dashboard

| Risk ID | Description | Probability | Impact | Score | Owner | Status |
|---------|-------------|-------------|--------|-------|-------|--------|
| RISK-001 | vLLM SPOF | M | H | 6 | Eng Lead | Open |
| RISK-002 | Quality Regression | L | H | 3 | QA Lead | Open |
| RISK-003 | GPU OOM | M | M | 4 | SRE Lead | Open |
| RISK-004 | Network Latency | L | M | 2 | Platform | Open |
| RISK-005 | Timeline Overrun | M | L | 2 | PM | Open |
| RISK-006 | Security Vuln | L | H | 3 | Security | Open |
| RISK-007 | Circuit Breaker | M | M | 4 | Eng Lead | Open |
| RISK-008 | Cost Overrun | M | L | 2 | Finance | Open |
| RISK-009 | Knowledge Gap | M | M | 4 | Ops Lead | Open |
| RISK-010 | Rollback Complexity | L | H | 3 | Deploy Lead | Open |

**High-Priority Risks** (Score ≥6): RISK-001 (vLLM SPOF)

**Action Item**: Evaluate 2-replica vLLM deployment for production

---

## Operational Readiness Review (ORR)

### ORR Checklist

#### 1. Documentation Readiness

- [ ] **Architecture Documentation** (`docs/gpu-microservices.md`)
  - Updated with split-container section
  - Diagrams accurate
  - Reviewed and approved

- [ ] **Deployment Guide** (`docs/devops/vllm-deployment.md`)
  - Step-by-step deployment instructions
  - Tested on staging
  - Kubernetes manifests validated

- [ ] **Operations Runbooks** (`OPERATIONS.md`)
  - vLLM server restart procedure tested
  - High latency troubleshooting tested
  - Circuit breaker recovery tested
  - Incident response procedures documented

- [ ] **Security Documentation** (`SECURITY.md`)
  - Threat model completed
  - NetworkPolicy tested
  - Penetration test results reviewed

- [ ] **Training Materials**
  - Operations training deck complete
  - User training webinar recorded
  - Runbook videos published

---

#### 2. Infrastructure Readiness

- [ ] **Staging Environment**
  - vLLM server deployed and stable (48+ hours)
  - 8 workers deployed and processing PDFs
  - Monitoring/alerting functional
  - All E2E tests passing

- [ ] **Production Environment**
  - GPU node provisioned (RTX 5090)
  - Kubernetes cluster ready
  - Persistent volumes created
  - Secrets configured (API keys, tokens)
  - NetworkPolicies deployed

- [ ] **Backup & DR**
  - PersistentVolume snapshot configured
  - Backup procedure tested
  - Disaster recovery runbook validated

---

#### 3. Monitoring & Alerting Readiness

- [ ] **Metrics Collection**
  - Prometheus scraping vLLM metrics
  - Prometheus scraping worker metrics
  - Custom metrics for circuit breaker, throughput

- [ ] **Dashboards**
  - vLLM server dashboard created (GPU, latency, queue depth)
  - Worker dashboard created (HTTP client metrics)
  - End-to-end pipeline dashboard
  - SLO dashboard

- [ ] **Alerts**
  - vLLM server down alert configured
  - High latency alert configured
  - Circuit breaker stuck open alert
  - GPU OOM alert
  - All alerts tested (intentionally triggered)

- [ ] **On-Call**
  - PagerDuty integration configured
  - On-call rotation schedule updated
  - Escalation policy defined

---

#### 4. Testing Readiness

- [ ] **Unit Tests**: 100% passing (25+ tests)
- [ ] **Integration Tests**: 100% passing (15+ tests)
- [ ] **E2E Tests**: ≥98% passing (100 PDFs)
- [ ] **Performance Tests**: Throughput ≥60 PDFs/hour, P95 latency <90s
- [ ] **Chaos Tests**: vLLM restart recovery validated
- [ ] **Security Tests**: Penetration test passed
- [ ] **Quality Tests**: ≥95% output match with baseline

---

#### 5. Team Readiness

- [ ] **Operations Team**
  - Training completed (2-hour workshop)
  - Runbooks reviewed
  - Rollback procedure practiced
  - On-call handoff completed

- [ ] **Engineering Team**
  - Implementation complete
  - Code reviewed and merged
  - Knowledge transfer to ops completed

- [ ] **Support Team**
  - User-facing changes communicated
  - Support runbook updated
  - Escalation paths defined

---

#### 6. Stakeholder Readiness

- [ ] **Executive Approval**: Deployment plan approved
- [ ] **Product Approval**: Feature parity validated, performance improvements confirmed
- [ ] **Finance Approval**: Budget approved
- [ ] **Security Approval**: Security review passed
- [ ] **Legal/Compliance** (if applicable): HIPAA compliance validated

---

#### 7. Communication Readiness

- [ ] **Pre-Deployment Announcement**: Sent to all stakeholders (Day -3)
- [ ] **Status Update Template**: Prepared for daily updates
- [ ] **Incident Communication Plan**: Templates ready
- [ ] **Stakeholder Matrix**: Up-to-date contact list

---

#### 8. Deployment Readiness

- [ ] **Deployment Procedure**: Step-by-step plan documented (DEPLOYMENT_STRATEGY.md)
- [ ] **Rollback Procedure**: Tested in staging
- [ ] **Go/No-Go Criteria**: Defined with quantified metrics
- [ ] **Deployment Timeline**: Communicated to stakeholders
- [ ] **Change Advisory Board** (if required): Approval obtained

---

### Production Readiness Criteria

**Must-Have (Blockers)**:

- [ ] All "Documentation Readiness" items complete
- [ ] All "Infrastructure Readiness" items complete
- [ ] All "Monitoring & Alerting Readiness" items complete
- [ ] All "Testing Readiness" items complete (98% pass rate minimum)
- [ ] Operations team training complete
- [ ] Security approval obtained

**Should-Have (Non-Blockers)**:

- Team readiness items (can be completed during rollout)
- Communication items (can be adjusted on-the-fly)

**Go/No-Go Decision**: If all "Must-Have" items are complete, proceed with deployment. If any blockers remain, delay deployment until resolved.

---

## ORR Meeting Agenda

### ORR Review Meeting (Scheduled: Day -3)

**Participants**:

- Engineering Lead (Chair)
- Operations Lead
- SRE Lead
- Security Lead
- QA Lead
- Product Manager

**Agenda** (2 hours):

1. **Welcome & Objectives** (10 min)
   - Purpose of ORR
   - Decision: Go/No-Go for deployment

2. **Documentation Review** (20 min)
   - Walk through updated documentation
   - Gaps identified and resolved?

3. **Infrastructure Walkthrough** (20 min)
   - Staging environment demo
   - Production environment readiness confirmation

4. **Monitoring & Alerting Demo** (20 min)
   - Live Grafana dashboard walkthrough
   - Alert triggering demo

5. **Testing Results** (20 min)
   - Test pass rates presented
   - Quality comparison results
   - Performance benchmark results

6. **Team & Operational Readiness** (15 min)
   - Training completion status
   - On-call readiness
   - Runbook validation

7. **Risk Review** (10 min)
   - Walk through risk register
   - Mitigation plans confirmed

8. **Go/No-Go Decision** (15 min)
   - Poll each participant
   - Document decision and rationale
   - If Go: Confirm deployment start time
   - If No-Go: Define blocking items and re-ORR date

**Deliverable**: ORR Sign-Off Document (signatures from all participants)

---

## Quality Gates

### Gate 1: Pre-Implementation (Week 0)

**Criteria**:

- [ ] Design approved by Engineering Lead
- [ ] Tasks.md reviewed and estimated
- [ ] Implementation team assigned
- [ ] Timeline communicated to stakeholders

**Decision**: Proceed with implementation?

---

### Gate 2: Mid-Implementation (Week 3)

**Criteria**:

- [ ] 50% of tasks complete
- [ ] No major blockers
- [ ] Staging environment functional
- [ ] Unit tests passing

**Decision**: Continue implementation on schedule?

---

### Gate 3: Pre-Deployment (Week 5)

**Criteria**:

- [ ] All tasks complete
- [ ] ORR checklist 100% complete
- [ ] Security approval
- [ ] Stakeholder approvals

**Decision**: Proceed with deployment?

---

### Gate 4: Post-Deployment (Week 6)

**Criteria**:

- [ ] 100% rollout successful
- [ ] SLOs met for 7 days
- [ ] No critical incidents
- [ ] Operational team confident

**Decision**: Decommission old deployment?

---

## Defect Management

### Defect Severity Classification

| Severity | Criteria | Response Time | Example |
|----------|----------|---------------|---------|
| **Sev1 (Critical)** | Service down, data loss | Immediate | vLLM server crash loop |
| **Sev2 (High)** | Major feature broken | <4 hours | Workers can't connect to vLLM |
| **Sev3 (Medium)** | Minor feature broken | <1 day | Metrics not collected |
| **Sev4 (Low)** | Cosmetic, docs | <1 week | Typo in runbook |

### Defect Workflow

1. **Detection**: Automated tests, monitoring, user reports
2. **Triage**: Assign severity, owner
3. **Fix**: Develop fix, test in staging
4. **Deploy**: Rollout fix to production
5. **Verify**: Confirm fix effective
6. **Close**: Update documentation, close ticket

**Defect Tracking**: Jira project `MINERU-SPLIT`

### Acceptance Criteria for Deployment Phases

#### Phase 1 (10% Rollout) Acceptance Criteria

- [ ] 96+ hours stable operation (4 days)
- [ ] Error rate ≤baseline +5%
- [ ] P95 latency ≤baseline +10%
- [ ] No Sev1/Sev2 incidents
- [ ] Quality match ≥95% (sample 50 PDFs)

**Decision**: Proceed to Phase 2 (50%)?

#### Phase 2 (50% Rollout) Acceptance Criteria

- [ ] 7 days stable operation
- [ ] Error rate ≤baseline +2%
- [ ] P95 latency ≤baseline +5%
- [ ] Throughput ≥baseline +15%
- [ ] No Sev1 incidents
- [ ] Quality match ≥95% (sample 100 PDFs)

**Decision**: Proceed to Phase 3 (100%)?

#### Phase 3 (100% Rollout) Acceptance Criteria

- [ ] 10 days stable operation
- [ ] SLOs met (availability, latency, throughput, error rate)
- [ ] Operations team confident (no escalations needed)
- [ ] Cost within budget

**Decision**: Decommission old deployment?

---

**Document Control**:

- **Version**: 1.0
- **Last Updated**: 2025-10-08
- **Owner**: Project Manager
- **Approvers**: Engineering Lead, Operations Lead, Security Lead
