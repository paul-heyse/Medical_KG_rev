# Training and Communication Plan: MinerU vLLM Split-Container

**Change ID**: `update-mineru-vllm-split-container`
**Version**: 1.0
**Date**: 2025-10-08

---

## Stakeholder Communication Matrix

| Stakeholder | Role | Information Needs | Frequency | Channel | Owner |
|-------------|------|-------------------|-----------|---------|-------|
| **Executive Team** | Decision makers | Business impact, costs, risks | Pre-deployment + Post-deployment | Email summary | Deployment Lead |
| **Engineering Team** | Implementers | Technical details, blockers, changes | Daily during rollout | Slack #medical-kg-deploy | Engineering Lead |
| **Operations Team** | Operators | Runbooks, monitoring, on-call changes | Weekly + Ad-hoc | Slack #medical-kg-ops | Operations Lead |
| **Product Team** | Feature owners | User impact, timeline, quality | Weekly | Slack #medical-kg-product | Product Manager |
| **Data Science Team** | End users | API changes, performance improvements | Pre-deployment + Training | Email + Training session | Product Manager |
| **Support Team** | User support | Known issues, escalation paths | Pre-deployment | Confluence + Training | Support Lead |
| **Security Team** | Security oversight | Architecture changes, compliance | Pre-deployment + Quarterly | Email + Review meetings | Security Lead |

---

## Communication Timeline

### Pre-Deployment Phase (Week -2 to Week 0)

**Week -2**: Initial Announcement

- **Audience**: All stakeholders
- **Content**: Architecture change overview, benefits, timeline
- **Channel**: Email + Slack announcement
- **Owner**: Deployment Lead

**Week -1**: Technical Deep Dive

- **Audience**: Engineering, Operations
- **Content**: Implementation details, deployment plan, runbooks
- **Channel**: Engineering meeting + documentation share
- **Owner**: Engineering Lead

**Day -3**: Go/No-Go Preparation

- **Audience**: Deployment team
- **Content**: Readiness checklist review, final preparations
- **Channel**: Video call + Slack
- **Owner**: Deployment Lead

### Deployment Phase (Week 1-3)

**Daily**: Status Updates

- **Audience**: Engineering, Operations, Product
- **Content**: Deployment progress, metrics, issues
- **Channel**: Slack #medical-kg-deploy
- **Owner**: Deployment Lead

**Weekly**: Executive Summary

- **Audience**: Executive team, Finance
- **Content**: Progress, risks, cost impact
- **Channel**: Email
- **Owner**: Engineering Lead

### Post-Deployment Phase (Week 4+)

**Week 4**: Completion Announcement

- **Audience**: All stakeholders
- **Content**: Success metrics, lessons learned, next steps
- **Channel**: Email + Slack
- **Owner**: Deployment Lead

**Week 6**: Retrospective

- **Audience**: Deployment team
- **Content**: What went well, what to improve
- **Channel**: Video call + retrospective document
- **Owner**: Engineering Lead

---

## Training Plan

### Training Audience Segmentation

#### Audience 1: Operations Engineers (Priority: Critical)

**Training Objectives**:

- Understand split-container architecture
- Execute vLLM server restart runbook
- Troubleshoot common issues (circuit breaker, latency)
- Use monitoring dashboards effectively

**Training Format**: 2-hour hands-on workshop

**Agenda**:

1. **Architecture Overview** (30 min):
   - Split-container pattern explained
   - Component diagram (vLLM server, workers, communication flow)
   - Benefits and tradeoffs

2. **Runbook Walkthroughs** (60 min):
   - Live demo: vLLM server restart procedure
   - Live demo: Troubleshooting high latency
   - Live demo: Circuit breaker recovery
   - Q&A

3. **Monitoring & Alerting** (30 min):
   - Grafana dashboard tour
   - Alert interpretation
   - Incident response workflow
   - Escalation procedures

**Materials**:

- Slide deck: `docs/training/mineru-split-container-ops-training.pdf`
- Runbooks: `OPERATIONS.md`
- Dashboard links: Grafana bookmarks

**Delivery**: Week 3 of deployment (before 100% rollout)

**Trainer**: Engineering Lead

**Assessment**: Post-training quiz (10 questions, 80% pass required)

---

#### Audience 2: Data Scientists (Priority: Medium)

**Training Objectives**:

- Understand performance improvements
- No API changes (transparent to users)
- Know who to contact for issues

**Training Format**: 30-minute webinar

**Agenda**:

1. **What's Changing** (10 min):
   - Architecture upgrade overview
   - Performance improvements: +20% throughput, faster startup
   - No action required from users

2. **What's Staying the Same** (10 min):
   - API unchanged
   - PDF processing quality maintained
   - Same endpoints, same workflows

3. **FAQ & Support** (10 min):
   - How to report issues
   - Support channels
   - Q&A

**Materials**:

- Slide deck: `docs/training/mineru-split-container-user-training.pdf`
- FAQ document: `docs/FAQ.md`

**Delivery**: Week 2 of deployment

**Trainer**: Product Manager

**Recording**: Available on Confluence

---

#### Audience 3: On-Call Engineers (Priority: Critical)

**Training Objectives**:

- Respond to new alert types
- Execute incident response procedures
- Understand escalation paths

**Training Format**: 1-hour technical deep dive

**Agenda**:

1. **New Alert Rules** (20 min):
   - vLLM server down
   - High latency
   - Circuit breaker stuck open
   - Alert severity classification

2. **Incident Response Drills** (30 min):
   - Simulated incident: vLLM server OOM
   - Walk through runbook execution
   - Practice communication protocol

3. **Escalation & Handoff** (10 min):
   - When to escalate
   - Who to escalate to
   - On-call handoff checklist

**Materials**:

- `OPERATIONS.md` (Incident Response Runbook section)
- On-call playbook: `docs/on-call/mineru-playbook.md`

**Delivery**: Week 2 of deployment (before Phase 2 50% rollout)

**Trainer**: SRE Lead

**Assessment**: Incident response drill (pass/fail)

---

### Training Materials Creation

#### Slide Decks

**Deck 1: Operations Training**

- **Slides**: 30-40 slides
- **Content**:
  - Architecture diagrams
  - Runbook screenshots
  - Dashboard screenshots
  - Decision trees (troubleshooting flowcharts)
- **Format**: PowerPoint/Google Slides
- **Creation**: Week 2 of implementation
- **Owner**: Engineering Lead

**Deck 2: User Training**

- **Slides**: 10-15 slides
- **Content**:
  - High-level architecture
  - Performance improvements
  - FAQ
- **Format**: PowerPoint/Google Slides
- **Creation**: Week 3 of implementation
- **Owner**: Product Manager

#### Video Recordings

**Video 1: vLLM Server Restart Runbook (5 min)**

- **Content**: Screen recording of runbook execution
- **Platform**: Loom or Confluence video
- **Owner**: Engineering Lead

**Video 2: Grafana Dashboard Tour (10 min)**

- **Content**: Walkthrough of key metrics, how to interpret
- **Platform**: Loom or Confluence video
- **Owner**: SRE Lead

**Video 3: User Webinar Recording**

- **Content**: Full user training webinar
- **Platform**: Zoom recording → Confluence
- **Owner**: Product Manager

#### Written Documentation

**Document 1: Architecture Decision Records (ADRs)**

- **Location**: `docs/architecture/decisions/`
- **Content**:
  - ADR-001: Split-container architecture (why, alternatives, decision)
  - ADR-002: vLLM OpenAI-compatible server (why not Ray Serve)
  - ADR-003: Circuit breaker pattern (failure thresholds chosen)
- **Format**: Markdown
- **Owner**: Engineering Lead

**Document 2: Developer Onboarding Updates**

- **Location**: `docs/ONBOARDING.md`
- **Updates**:
  - Add split-container architecture section
  - Update local development setup (vLLM server in Docker Compose)
  - Update troubleshooting guide
- **Owner**: Engineering Lead

**Document 3: API Documentation**

- **Location**: `docs/openapi.yaml`, `docs/schema.graphql`
- **Updates**: None (API unchanged, document non-change)
- **Owner**: Engineering Lead

---

## Knowledge Transfer Sessions

### Session 1: Engineering Handoff (Pre-Deployment)

**Participants**: Implementation team → Operations team

**Objectives**:

- Transfer implementation knowledge
- Review design decisions
- Walk through codebase changes

**Agenda** (2 hours):

1. Code walkthrough: VLLMClient, Circuit Breaker
2. Configuration review: Kubernetes manifests
3. Testing strategy review
4. Q&A

**Deliverables**:

- Meeting notes
- Confluence page with key learnings

**Schedule**: Week 4 of implementation (before deployment)

---

### Session 2: Operations Deep Dive (Post-Deployment)

**Participants**: Operations team → Support team

**Objectives**:

- Share operational learnings from first 2 weeks
- Update support escalation procedures

**Agenda** (1 hour):

1. Common issues encountered
2. Resolution patterns
3. When to escalate to engineering
4. Updated support runbook

**Deliverables**:

- Updated support runbook

**Schedule**: Week 6 (2 weeks after 100% rollout)

---

## Documentation Review Process

### Pre-Deployment Documentation Review

**Reviewers**: Engineering Lead, Operations Lead, Security Lead

**Documents to Review**:

- [ ] Architecture docs (`docs/gpu-microservices.md`)
- [ ] Deployment guide (`docs/devops/vllm-deployment.md`)
- [ ] Operations runbooks (`OPERATIONS.md`)
- [ ] Security documentation (`SECURITY.md`)
- [ ] Training materials (slide decks, videos)

**Review Checklist**:

- [ ] Technical accuracy verified
- [ ] Runbooks tested on staging
- [ ] Screenshots up-to-date
- [ ] No sensitive information (passwords, keys) exposed
- [ ] Links functional
- [ ] Formatting consistent

**Timeline**: Week 3 of implementation (1 week before deployment)

---

### Post-Deployment Documentation Updates

**Updates Based on Learnings** (Week 6):

- Update runbooks with real-world incident learnings
- Add FAQ entries from common questions
- Update troubleshooting guide with production issues
- Add performance tuning recommendations

**Owner**: Engineering Lead with input from Operations

---

## Communication Escalation

### Escalation Matrix

| Issue Severity | Initial Contact | Escalation (15 min) | Escalation (30 min) | Escalation (1 hour) |
|----------------|-----------------|---------------------|---------------------|---------------------|
| **Sev1** | On-call Engineer | Engineering Lead | VP Engineering | CTO |
| **Sev2** | On-call Engineer | Engineering Lead | VP Engineering | - |
| **Sev3** | Slack #medical-kg-deploy | Engineering Lead | - | - |

### Communication Templates (See DEPLOYMENT_STRATEGY.md)

Pre-written templates for:

- Pre-deployment announcement
- Daily status update
- Go/No-Go meeting agenda
- Incident notification
- Post-deployment success announcement

---

## Success Metrics

### Training Effectiveness

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Training Completion Rate** | 100% (critical audiences) | Attendance tracking |
| **Post-Training Assessment** | ≥80% pass rate | Quiz scores |
| **Runbook Usage** | ≥5 unique users in first month | Analytics on Confluence page |
| **Support Ticket Reduction** | <5 tickets related to split-container | Jira ticket tracking |

### Communication Effectiveness

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Stakeholder Satisfaction** | ≥4/5 rating | Post-deployment survey |
| **Incident Response Time** | <15 min (Sev2) | Incident log analysis |
| **Documentation Clarity** | ≥4/5 rating | Doc feedback form |

---

**Document Control**:

- **Version**: 1.0
- **Last Updated**: 2025-10-08
- **Owner**: Deployment Lead
- **Approvers**: Engineering Lead, Product Manager, Operations Lead
