# Deployment Strategy: MinerU vLLM Split-Container Architecture

**Change ID**: `update-mineru-vllm-split-container`
**Version**: 1.0
**Date**: 2025-10-08
**Status**: Ready for Review

---

## Table of Contents

1. [Pre-Deployment Readiness](#pre-deployment-readiness)
2. [Phased Rollout Plan](#phased-rollout-plan)
3. [Go/No-Go Decision Criteria](#gono-go-decision-criteria)
4. [Rollback Procedures](#rollback-procedures)
5. [Communication Plan](#communication-plan)
6. [Post-Deployment Validation](#post-deployment-validation)
7. [Deployment Approval Workflow](#deployment-approval-workflow)

---

## Pre-Deployment Readiness

### Infrastructure Prerequisites

#### Staging Environment

- [ ] **vLLM Server**: Deployed to staging, health checks passing for 48+ hours
- [ ] **MinerU Workers**: 8 worker pods deployed, connecting successfully to vLLM
- [ ] **Kubernetes Resources**: Deployments, Services, NetworkPolicies validated
- [ ] **Monitoring**: Prometheus scraping vLLM and worker metrics
- [ ] **Alerting**: All alert rules deployed and tested (trigger test alerts)
- [ ] **Grafana Dashboards**: vLLM server, worker client, E2E pipeline dashboards imported

#### Production Environment

- [ ] **GPU Node Pool**: RTX 5090 nodes available with NVIDIA Container Toolkit
- [ ] **Kubernetes Cluster**: Version 1.27+ with GPU scheduling enabled
- [ ] **Persistent Storage**: Hugging Face cache volume for vLLM model storage
- [ ] **Network Policies**: Deployed and validated (test connectivity)
- [ ] **Secrets Management**: vLLM server config secrets created
- [ ] **Service Mesh** (if applicable): Istio/Linkerd configured for vLLM traffic

### Testing Completion

- [ ] **Unit Tests**: All 25+ unit tests passing (100% coverage for new code)
- [ ] **Integration Tests**: 15+ integration tests passing (worker-vLLM connectivity)
- [ ] **E2E Tests**: 100 PDFs processed successfully end-to-end
- [ ] **Performance Tests**: Throughput ≥48 PDFs/hour, P95 latency <90s
- [ ] **Chaos Tests**: vLLM server restart recovery validated
- [ ] **Quality Tests**: Output comparison shows ≥95% match with baseline
- [ ] **Load Tests**: System handles 12 concurrent workers without degradation

### Documentation Completion

- [ ] **Architecture Docs**: `docs/gpu-microservices.md` updated with split-container section
- [ ] **Deployment Guide**: `docs/devops/vllm-deployment.md` complete and reviewed
- [ ] **Runbooks**: vLLM server restart, troubleshooting guides complete
- [ ] **API Docs**: OpenAPI spec updated (if API changes)
- [ ] **Training Materials**: Slide deck and demo prepared
- [ ] **Migration Guide**: `docs/migrations/mineru-split-container-migration.md` complete

### Security & Compliance

- [ ] **Threat Model**: STRIDE analysis completed for split-container architecture
- [ ] **Security Scan**: Container images scanned with Trivy/Snyk (no high/critical vulnerabilities)
- [ ] **Penetration Test**: Security team completed pentest of vLLM/worker communication
- [ ] **Network Policy Test**: Validated isolation (only workers can access vLLM)
- [ ] **Secrets Rotation**: Tested secret rotation without downtime
- [ ] **Audit Logs**: Verified vLLM API calls logged with tenant_id, correlation_id
- [ ] **Compliance Review**: Legal/compliance sign-off (if HIPAA/SOC 2 applicable)

### Operational Readiness

- [ ] **Monitoring**: Grafana dashboards reviewed with ops team, alerts tuned
- [ ] **Runbooks**: Ops team trained on vLLM server restart, troubleshooting procedures
- [ ] **On-Call**: On-call engineers briefed on new architecture, escalation paths defined
- [ ] **Capacity Plan**: Validated capacity for 100% traffic (8-12 workers + 1 vLLM server)
- [ ] **Backup Plan**: Disaster recovery tested (restore vLLM server from snapshot)
- [ ] **Support Tickets**: Confluence/Jira space created for split-container issues

### Stakeholder Approvals

- [ ] **Engineering Lead**: Approved implementation quality and test results
- [ ] **Product Manager**: Approved feature parity and performance improvements
- [ ] **Operations Manager**: Approved operational readiness and runbooks
- [ ] **Security Lead**: Approved security posture and compliance
- [ ] **Finance**: Approved budget impact (GPU resource changes)

---

## Phased Rollout Plan

### Phase 0: Production Preparation (Day -3 to Day 0)

**Objective**: Deploy infrastructure without routing traffic

**Timeline**: 3 days before rollout start

**Activities**:

1. **Day -3**:
   - Deploy vLLM server to production (0% traffic, not in service mesh)
   - Validate health checks, model loading, metrics collection
   - Deploy updated MinerU workers with feature flag OFF (monolithic backend)
   - Smoke test: Manually trigger 10 PDFs through new workers (monolithic mode)

2. **Day -2**:
   - Monitor vLLM server for 24 hours (GPU memory, latency, stability)
   - Verify Prometheus metrics collection and Grafana dashboards
   - Test alert firing (intentionally trigger alerts, verify notifications)
   - Review runbooks with on-call engineers

3. **Day -1**:
   - **Go/No-Go Meeting #1**: Review readiness checklist, decide to proceed
   - Freeze code (no new changes except critical fixes)
   - Send stakeholder communication: "Deployment starts tomorrow"
   - On-call team briefed and ready

### Phase 1: 10% Traffic (Day 1-4)

**Objective**: Validate split-container with small user subset

**Timeline**: Days 1-4 (96 hours)

**Activities**:

**Day 1 (Morning)**:

- **09:00**: Enable split-container for 10% of workers (feature flag or pod labels)
- **09:30**: Validate traffic routing (check logs, metrics show vLLM requests)
- **10:00**: Monitor for 1 hour (error rates, latency, GPU utilization)
- **11:00**: Send status update: "10% rollout successful, monitoring"

**Day 1 (Afternoon/Evening)**:

- **Continuous Monitoring**: Every 4 hours check dashboards, review alerts
- **Incident Response**: On-call engineer monitoring Slack channel

**Day 2-4**:

- **Daily**: Morning stand-up to review metrics from previous 24 hours
- **Daily**: Compare 10% cohort vs 90% baseline (error rate, latency, quality)
- **Metrics to Track**:
  - Error rate delta: <+5% acceptable
  - P95 latency delta: <+10% acceptable
  - vLLM server stability: No crashes, GPU memory stable
  - Worker circuit breaker events: <5 opens per day
  - Quality regression: Sample 50 PDFs, compare outputs

**Day 4 (End of Phase)**:

- **Go/No-Go Meeting #2**: Review 96-hour metrics, decide to proceed to 50%

### Phase 2: 50% Traffic (Day 5-11)

**Objective**: Validate split-container at half capacity

**Timeline**: Days 5-11 (168 hours / 7 days)

**Activities**:

**Day 5 (Morning)**:

- **09:00**: Increase to 50% traffic (scale split-container workers to 50%)
- **09:30**: Monitor for spikes in latency or error rate
- **10:00**: Validate vLLM server handles increased load (GPU utilization target: 85%)
- **11:00**: Send status update: "50% rollout in progress"

**Day 5-7** (First 72 hours):

- **Intensive Monitoring**: Check dashboards every 2 hours during business hours
- **Automated Alerts**: Rely on alerting system for off-hours monitoring
- **A/B Testing**: Compare 50% split-container vs 50% monolithic cohorts
  - Side-by-side dashboards showing both cohorts
  - Statistical significance testing on latency distributions

**Day 8-11** (Steady State):

- **Reduced Monitoring**: Daily check-ins, rely on automated alerting
- **Performance Tuning**: Adjust vLLM batch size, worker connection pools based on observed patterns
- **Capacity Validation**: Simulate 100% load (scale split-container workers to 12, measure impact)

**Day 11 (End of Phase)**:

- **Go/No-Go Meeting #3**: Review 7-day metrics, decide to proceed to 100%
- **Metrics Review**:
  - Error rate: Must be ≤baseline +2%
  - P95 latency: Must be ≤baseline +5%
  - vLLM server uptime: Must be >99.9%
  - Quality regression: <1% (sample 100 PDFs)

### Phase 3: 100% Traffic (Day 12+)

**Objective**: Complete migration to split-container

**Timeline**: Day 12 onwards (ongoing)

**Activities**:

**Day 12 (Morning)**:

- **09:00**: Increase to 100% traffic (all workers use split-container)
- **09:30**: Monitor closely for 2 hours (highest risk period)
- **10:00**: Validate all workers connected to vLLM server
- **11:00**: Send status update: "100% rollout complete, monitoring"

**Day 12 (Full Day)**:

- **Intensive Monitoring**: Every hour during business hours
- **Incident Response**: Two on-call engineers available

**Day 13-18** (First Week at 100%):

- **Daily Stand-ups**: Review previous 24-hour metrics
- **Weekly Metrics Report**: Send to stakeholders (error rate, latency, throughput, GPU utilization)
- **Fine-Tuning**: Optimize based on production traffic patterns

**Day 19+ (Steady State)**:

- **Standard Monitoring**: Automated alerting, weekly reviews
- **Performance Optimization**: Continue tuning (see Phase 10 in tasks.md)

### Phase 4: Cleanup (Day 22+)

**Objective**: Decommission old monolithic deployment

**Timeline**: After 10 days of stable 100% traffic

**Activities**:

- **Day 22**:
  - **Decommission Check**: Confirm 10 days stable, no rollback needed
  - Remove feature flag from configuration
  - Delete old monolithic worker deployment
  - Clean up old Docker images from registry (keep last 2 for emergency rollback)
  - Archive migration documentation
  - Send final status update: "Split-container migration complete"

---

## Go/No-Go Decision Criteria

### Decision Gate 1: Pre-Deployment Readiness (Day -1)

**Decision**: Proceed with Phase 1 (10% rollout)?

**Criteria** (ALL must be met):

| Criteria | Threshold | Measurement |
|----------|-----------|-------------|
| **Staging Stability** | 48+ hours uptime | vLLM server in staging, no crashes |
| **Test Pass Rate** | 100% | All unit, integration, E2E tests passing |
| **Performance Baseline** | Established | Staging throughput ≥48 PDFs/hr |
| **Documentation Complete** | 100% | All runbooks, guides reviewed and approved |
| **Security Sign-off** | Approved | Security team completed review |
| **Stakeholder Approval** | All approved | Engineering, Product, Ops, Security leads signed off |

**Go Decision**: All criteria met → Proceed to Phase 1
**No-Go Decision**: Any criterion not met → Delay deployment, remediate gaps

### Decision Gate 2: 10% to 50% (Day 4)

**Decision**: Increase from 10% to 50%?

**Criteria** (ALL must be met):

| Criteria | Threshold | Measurement |
|----------|-----------|-------------|
| **Error Rate Delta** | <+5% | Compare 10% cohort vs 90% baseline |
| **P95 Latency Delta** | <+10% | Compare 10% cohort vs 90% baseline |
| **vLLM Server Stability** | 0 crashes | Monitor for OOM, GPU errors |
| **Worker Circuit Breaker** | <5 opens/day | Count circuit breaker open events |
| **Quality Regression** | <1% | Sample 50 PDFs, compare structure/content |
| **Critical Incidents** | 0 | No Sev1/Sev2 incidents related to split-container |

**Go Decision**: All criteria met → Proceed to Phase 2
**No-Go Decision**: Any criterion exceeded → Rollback to 0%, investigate root cause

### Decision Gate 3: 50% to 100% (Day 11)

**Decision**: Increase from 50% to 100%?

**Criteria** (ALL must be met):

| Criteria | Threshold | Measurement |
|----------|-----------|-------------|
| **Error Rate** | ≤baseline +2% | 7-day average comparison |
| **P95 Latency** | ≤baseline +5% | 7-day P95 comparison |
| **Throughput** | ≥baseline +15% | 7-day average throughput |
| **vLLM Server Uptime** | >99.9% | 7 days = 168 hours, max 10 min downtime |
| **GPU Utilization** | >80% | Average GPU utilization over 7 days |
| **Quality Regression** | <1% | Sample 100 PDFs, automated comparison |
| **Stakeholder Confidence** | High | Poll key stakeholders for concerns |

**Go Decision**: All criteria met → Proceed to Phase 3
**No-Go Decision**: Any criterion exceeded → Hold at 50%, investigate and remediate

### Decision Gate 4: Cleanup (Day 22)

**Decision**: Decommission old monolithic deployment?

**Criteria** (ALL must be met):

| Criteria | Threshold | Measurement |
|----------|-----------|-------------|
| **100% Stability** | 10 days stable | No critical incidents, <5 alerts fired |
| **Error Rate** | ≤baseline | 10-day average at or below original baseline |
| **Latency SLO** | Met | P95 <90s for 10 consecutive days |
| **No Rollback Requests** | 0 | Engineering/Ops not requesting rollback capability |
| **Cost Savings Realized** | Confirmed | Finance confirms GPU cost savings |

**Go Decision**: All criteria met → Decommission old deployment
**No-Go Decision**: Any criterion not met → Keep old deployment available for 30 more days

---

## Rollback Procedures

### Rollback Triggers

**Automatic Rollback** (Immediate, no human approval):

- **Never**: We do not implement automatic rollback for this change (too complex)

**Manual Rollback** (Ops/Engineering decision):

| Trigger | Severity | Action | Timeline |
|---------|----------|--------|----------|
| **Error Rate Spike** | Error rate >baseline +20% for 15 minutes | Rollback | Immediate |
| **Latency Spike** | P95 >120s for 15 minutes | Rollback | Immediate |
| **vLLM Server Crash Loop** | vLLM restarts >3 times in 30 minutes | Rollback | Immediate |
| **Quality Regression** | >5% of PDFs show structural errors | Rollback | Within 4 hours |
| **Security Incident** | Unauthorized access or data leak detected | Rollback | Immediate |
| **Critical Bug** | Data loss or corruption detected | Rollback | Immediate |

### Rollback Procedure: Feature Flag Method (Fastest)

**Time Required**: 5-10 minutes
**Applicable To**: Phase 1 (10%), Phase 2 (50%)

**Steps**:

1. **Decision** (2 min):
   - Incident Commander (IC) declares rollback decision
   - Notify stakeholders in `#medical-kg-incidents` Slack channel

2. **Execute Rollback** (3 min):

   ```bash
   # Option A: Update ConfigMap (if using ConfigMap for feature flag)
   kubectl edit configmap mineru-config -n medical-kg
   # Change: deployment_mode: "split-container" → "monolithic"
   # Save and exit

   # Restart workers to pick up new config
   kubectl rollout restart deployment/mineru-workers -n medical-kg

   # Option B: Update Deployment directly (if using env var)
   kubectl set env deployment/mineru-workers DEPLOYMENT_MODE=monolithic -n medical-kg
   # This automatically triggers rolling restart
   ```

3. **Validation** (5 min):
   - Watch rollout status: `kubectl rollout status deployment/mineru-workers -n medical-kg`
   - Check worker logs: Workers should log "Using monolithic backend (in-process vLLM)"
   - Check metrics: Error rate should decrease within 2-3 minutes
   - Test: Manually trigger 5 PDFs, verify processing works

4. **Post-Rollback** (immediate):
   - **Communication**: Send status update to stakeholders
   - **Incident Report**: Create incident post-mortem document
   - **Investigation**: Analyze logs, metrics to determine root cause
   - **Fix**: Develop fix in staging, re-test before next deployment attempt

### Rollback Procedure: Image Rollback Method (Comprehensive)

**Time Required**: 10-15 minutes
**Applicable To**: All phases, if feature flag doesn't work

**Steps**:

1. **Decision** (2 min):
   - Same as feature flag method

2. **Execute Rollback** (8 min):

   ```bash
   # Rollback workers to previous image
   kubectl set image deployment/mineru-workers \
     mineru-worker=ghcr.io/your-org/mineru-worker:monolithic \
     -n medical-kg

   # Watch rollout
   kubectl rollout status deployment/mineru-workers -n medical-kg

   # Delete vLLM server (optional, can leave running)
   kubectl scale deployment/vllm-server --replicas=0 -n medical-kg
   ```

3. **Validation** (5 min):
   - Same as feature flag method
   - Additionally verify: Old monolithic image is running (`kubectl describe pod`)

4. **Post-Rollback**:
   - Same as feature flag method

### Rollback Procedure: Full Environment Rollback (Nuclear Option)

**Time Required**: 30-45 minutes
**Applicable To**: Catastrophic failure, data corruption

**Steps**:

1. **Decision** (5 min):
   - Incident Commander + Engineering Lead approval required
   - Notify executive team

2. **Execute Rollback** (30 min):

   ```bash
   # Rollback entire namespace to previous state using GitOps (Argo CD / Flux)
   argocd app rollback medical-kg-prod --revision <previous-revision>

   # OR manual rollback
   kubectl delete -f ops/k8s/overlays/production/vllm-server/
   kubectl rollout undo deployment/mineru-workers -n medical-kg

   # Restore database state if data corruption detected (nuclear option)
   # 1. Stop all workers
   kubectl scale deployment/mineru-workers --replicas=0 -n medical-kg
   # 2. Restore Neo4j from backup
   # 3. Restore OpenSearch indices from snapshot
   # 4. Restart workers
   ```

3. **Validation** (10 min):
   - Full smoke test: Process 20 PDFs end-to-end
   - Verify data integrity: Check Neo4j, OpenSearch indices
   - Verify API endpoints: Test REST, GraphQL, gRPC

4. **Post-Rollback**:
   - **Incident Post-Mortem**: Required within 24 hours
   - **Executive Brief**: Update exec team on failure and remediation plan
   - **Hold Period**: No new deployments for 1 week, focus on investigation and fixes

### Post-Rollback Actions (All Rollback Types)

**Immediate** (within 1 hour):

- [ ] Rollback confirmed successful (system metrics back to normal)
- [ ] Incident channel updated with status
- [ ] On-call rotation notified

**Within 24 Hours**:

- [ ] Incident post-mortem document created
- [ ] Root cause analysis completed
- [ ] Fix identified and implemented in staging
- [ ] Stakeholders notified of fix plan

**Within 1 Week**:

- [ ] Fix validated in staging (48+ hours stable)
- [ ] Re-deployment plan created
- [ ] Approval obtained for re-deployment
- [ ] Retrospective meeting held with team

---

## Communication Plan

### Stakeholder Matrix

| Stakeholder | Role | Communication Needs | Frequency | Channel |
|-------------|------|---------------------|-----------|---------|
| **Executive Team** | Decision makers | High-level status, risk updates | Weekly | Email summary |
| **Engineering Team** | Implementers | Technical details, blockers | Daily | Slack #medical-kg-deploy |
| **Product Team** | Feature owners | User impact, quality metrics | Weekly | Slack #medical-kg-product |
| **Operations Team** | Operators | Deployment schedule, runbooks | Daily (during rollout) | Slack #medical-kg-ops |
| **Security Team** | Security oversight | Security testing, approvals | Pre-deployment + ad-hoc | Email + Slack |
| **Finance Team** | Budget owners | Cost impact | Pre-deployment + post-deployment | Email |
| **Support Team** | User-facing | Known issues, escalation | Pre-deployment + weekly | Confluence + Slack |

### Communication Templates

#### Pre-Deployment Announcement (Day -3)

**Subject**: [Medical_KG] MinerU Split-Container Deployment Starting [DATE]

**To**: All Stakeholders

**Template**:

```
Hi team,

We're beginning the deployment of the MinerU vLLM split-container architecture starting [DATE].

**What's Changing**:
- MinerU GPU service transitioning to split-container architecture
- Improved GPU utilization (+25%), faster worker startup (12x), higher throughput (+20-30%)

**Timeline**:
- Day -3 to 0: Infrastructure deployment (no user impact)
- Day 1-4: 10% traffic rollout (monitoring)
- Day 5-11: 50% traffic rollout (validation)
- Day 12+: 100% traffic rollout (full migration)

**Expected Impact**:
- No user-facing changes
- Potential for brief latency spikes during rollout (monitoring closely)
- Rollback plan in place if issues detected

**Communication Plan**:
- Daily updates in #medical-kg-deploy during rollout
- Immediate notification if rollback needed
- Weekly summary to exec team

**Questions**: Reply to this thread or ping @engineering-lead in Slack.

Thanks,
[Deployment Lead]
```

#### Daily Status Update (During Rollout)

**Subject**: [Medical_KG] Day [N] Status: [Phase] at [%] Traffic

**To**: Engineering, Operations, Product Teams

**Template**:

```
**Deployment Status - Day [N]**

**Current Phase**: [Phase 1/2/3] - [%] traffic on split-container

**Metrics (Last 24 Hours)**:
- Error Rate: [X]% (baseline: [Y]%, delta: [Z]%)
- P95 Latency: [X]s (baseline: [Y]s, delta: [Z]%)
- Throughput: [X] PDFs/hour (baseline: [Y], delta: [Z]%)
- vLLM Server Uptime: [X]%
- Critical Incidents: [N]

**Status**: 🟢 GREEN / 🟡 YELLOW / 🔴 RED

**Issues/Risks**:
- [List any issues or "None"]

**Next Steps**:
- [What happens next]

**Go/No-Go Decision**: [Date/Time] for next phase

Dashboard: [Link to Grafana dashboard]

Questions? #medical-kg-deploy
```

#### Go/No-Go Meeting Agenda

**Subject**: [Medical_KG] Go/No-Go: [Phase] to [Phase] Deployment

**Attendees**: Engineering Lead, Product Manager, Operations Manager, On-Call Engineer

**Agenda Template**:

```
**Go/No-Go Decision: Proceed from [Phase X] to [Phase Y]?**

**Metrics Review** (Present dashboard):
1. Error Rate: [Status vs criteria]
2. Latency: [Status vs criteria]
3. Throughput: [Status vs criteria]
4. Stability: [Status vs criteria]
5. Quality: [Status vs criteria]

**Incident Review**:
- Critical incidents: [Count and descriptions]
- Resolved issues: [Count and resolutions]

**Risk Assessment**:
- Known risks for next phase: [List]
- Mitigation plans: [List]

**Stakeholder Readiness**:
- Operations: Ready/Not Ready [Comments]
- Product: Ready/Not Ready [Comments]
- Engineering: Ready/Not Ready [Comments]

**Decision**:
- [ ] GO: Proceed to [Phase Y]
- [ ] NO-GO: Hold at [Phase X], investigate [issues]
- [ ] ROLLBACK: Revert to [Phase X-1], critical issue detected

**Action Items**:
- [Who] [What] [When]

**Next Meeting**: [Date/Time] for [Phase Y] to [Phase Z] decision
```

#### Incident Notification (Rollback Needed)

**Subject**: [URGENT] [Medical_KG] Rollback Initiated - [Reason]

**To**: All Stakeholders + On-Call

**Template**:

```
🚨 URGENT: Rollback in progress

**Reason**: [Brief description of issue]

**Impact**: [User-facing impact, if any]

**Actions**:
- Rollback initiated at [TIME] by [IC]
- Expected completion: [TIME] (10-15 minutes)
- Root cause investigation in progress

**Status Updates**: Every 15 minutes in #medical-kg-incidents

**Incident Channel**: #medical-kg-incidents
**Incident Commander**: @[name]

We'll provide updates as the situation evolves.

[Incident Commander]
```

#### Post-Deployment Success (Day 22)

**Subject**: [Medical_KG] MinerU Split-Container Migration Complete ✅

**To**: All Stakeholders

**Template**:

```
Hi team,

The MinerU vLLM split-container migration is complete! 🎉

**Results**:
- ✅ 100% traffic migrated successfully
- ✅ 10 days stable operation at 100%
- ✅ Performance improvements realized:
  - Throughput: +[X]% (baseline: [Y] PDFs/hr, now: [Z] PDFs/hr)
  - GPU Utilization: +[X]% (baseline: [Y]%, now: [Z]%)
  - Worker Startup: 12x faster ([X]s vs [Y]s)
- ✅ No critical incidents
- ✅ Error rate at or below baseline

**What's Next**:
- Decommissioning old monolithic deployment
- Continued performance optimization
- Post-deployment retrospective scheduled for [DATE]

**Documentation**:
- Architecture Docs: [Link]
- Runbooks: [Link]
- Performance Dashboard: [Link]

Thanks to everyone who contributed to this successful migration!

[Deployment Lead]
```

---

## Post-Deployment Validation

### Validation Checklist (Complete within 24 Hours of 100% Traffic)

#### Functional Validation

- [ ] **PDF Processing**: Process 100 test PDFs end-to-end, all succeed
- [ ] **Output Quality**: Sample 20 PDFs, compare outputs with baseline (≥95% match)
- [ ] **Worker Scaling**: Scale workers 8 → 12 → 8, verify smooth scaling
- [ ] **vLLM Server Restart**: Restart vLLM server, verify workers recover via circuit breaker
- [ ] **Network Partition**: Simulate network partition, verify circuit breaker opens and recovers

#### Performance Validation

- [ ] **Throughput**: Measure 24-hour average, confirm ≥baseline +15%
- [ ] **Latency**: Calculate P50, P95, P99, confirm P95 ≤baseline +5%
- [ ] **GPU Utilization**: Measure 24-hour average, confirm ≥80%
- [ ] **Worker Startup Time**: Measure 10 worker restarts, confirm <10s average
- [ ] **Error Rate**: Calculate 24-hour error rate, confirm ≤baseline

#### Observability Validation

- [ ] **Metrics Collection**: Verify all Prometheus metrics being scraped (vLLM + workers)
- [ ] **Dashboard Accuracy**: Spot-check Grafana dashboards match raw Prometheus queries
- [ ] **Alerting**: Trigger test alert, verify notification received in <2 minutes
- [ ] **Distributed Tracing**: Sample 10 traces, verify worker → vLLM spans present
- [ ] **Log Aggregation**: Verify vLLM and worker logs flowing to log aggregation system

#### Operational Validation

- [ ] **Runbook Accuracy**: Ops team executes vLLM restart runbook, validates steps
- [ ] **On-Call Handoff**: On-call engineer reviews dashboards, confirms understanding
- [ ] **Incident Response**: Simulate incident, verify escalation path works
- [ ] **Backup/DR**: Test restore from backup (in non-prod), confirm procedure works

#### Security Validation

- [ ] **Network Policy**: Attempt to access vLLM from non-worker pod, confirm blocked
- [ ] **API Authentication**: Attempt unauthenticated vLLM request, confirm rejected (if auth enabled)
- [ ] **Audit Logs**: Verify vLLM API calls logged with required fields
- [ ] **Secrets**: Verify secrets not exposed in logs, metrics, or dashboards
- [ ] **Container Security**: Re-scan images post-deployment, confirm no new vulnerabilities

### Long-Term Monitoring (First 30 Days)

**Week 1**:

- Daily dashboard review (morning stand-up)
- Weekly metrics report to stakeholders
- No major changes (feature freeze)

**Week 2-4**:

- 3x weekly dashboard review
- Bi-weekly metrics report
- Begin performance tuning (if needed)

**Day 30 Checkpoint**:

- [ ] Generate 30-day metrics report
- [ ] Calculate realized benefits vs projected
- [ ] Document lessons learned
- [ ] Plan retrospective meeting
- [ ] Update cost analysis with actual costs

---

## Deployment Approval Workflow

### Pre-Deployment Approval (Day -7)

**Approvers**:

1. **Engineering Lead**: Technical implementation quality
2. **Product Manager**: Feature parity and user impact
3. **Operations Manager**: Operational readiness
4. **Security Lead**: Security posture
5. **Finance Manager**: Budget impact (if significant)

**Approval Method**: Confluence page with checkbox approvals or Jira approval workflow

**Approval Criteria**: Pre-deployment readiness checklist 100% complete

### Phase Gate Approvals

**Approvers** (for each go/no-go decision):

1. **Incident Commander** (if during incident): Rollback decision authority
2. **Engineering Lead** OR **Operations Manager**: Phase progression authority

**Approval Method**: Go/No-Go meeting decision documented in meeting notes

**Approval Criteria**: Phase-specific go/no-go criteria all met

### Emergency Rollback Authority

**Immediate Rollback** (no approval needed):

- On-call engineer
- Incident Commander
- Engineering Lead
- Operations Manager

**Rationale**: In emergency situations, speed is critical. Post-rollback review will validate decision.

---

## Appendix: Quick Reference

### Key Contacts

| Role | Name | Slack | Phone | Responsibility |
|------|------|-------|-------|----------------|
| **Deployment Lead** | [Name] | @handle | [Phone] | Overall coordination |
| **Engineering Lead** | [Name] | @handle | [Phone] | Technical decisions |
| **Operations Manager** | [Name] | @handle | [Phone] | Operational readiness |
| **On-Call Engineer** | [Rotation] | @oncall | [PagerDuty] | Incident response |
| **Product Manager** | [Name] | @handle | [Phone] | User impact decisions |

### Critical Links

- **Deployment Dashboard**: [Grafana URL]
- **Incident Channel**: `#medical-kg-incidents`
- **Status Page**: [URL if exists]
- **Runbooks**: [Confluence space URL]
- **Rollback Procedure**: [This document, Section 4]

### Emergency Commands

```bash
# Quick rollback via feature flag
kubectl set env deployment/mineru-workers DEPLOYMENT_MODE=monolithic -n medical-kg

# Quick rollback via image
kubectl set image deployment/mineru-workers mineru-worker=ghcr.io/your-org/mineru-worker:monolithic -n medical-kg

# Scale down vLLM server (stop incoming requests)
kubectl scale deployment/vllm-server --replicas=0 -n medical-kg

# Check deployment status
kubectl rollout status deployment/mineru-workers -n medical-kg
kubectl rollout status deployment/vllm-server -n medical-kg
```

---

**Document Control**:

- **Version**: 1.0
- **Last Updated**: 2025-10-08
- **Next Review**: After deployment completion
- **Owner**: [Deployment Lead]
- **Approvers**: Engineering Lead, Operations Manager
