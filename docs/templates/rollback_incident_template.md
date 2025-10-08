# Rollback Incident Report Template

> Attach this template to any rollback drill or production rollback ticket.

## 1. Summary
- **Incident ID:**
- **Date / Time:**
- **Participants:** (On-call engineer, SRE, product owner)
- **Rollback Type:** (Automated / Manual / Canary)

## 2. Trigger Details
- **Trigger Source:** (Automated alert, manual decision, audit finding)
- **Trigger Metric / Observation:**
- **Threshold Breached:**
- **Related Dashboards:**

## 3. Timeline
| Time (UTC) | Event | Owner |
|------------|-------|-------|
|            |       |       |

## 4. Execution Steps
1. 
2. 
3. 

## 5. Outcome Metrics
- **Embedding Latency p95:** Before / After rollback
- **GPU Failure Rate:** Before / After
- **Tenant Isolation Alerts:** Count
- **Total Downtime:**

## 6. Communication
- **Stakeholder Updates Sent:** Yes/No (include links)
- **Customer Impact:**

## 7. Lessons Learned
- What went well?
- What needs improvement?
- Follow-up tasks created (link to tracking issues)

## 8. Post-Incident Review
- **Scheduled Time:** (Must be within 2 hours of rollback completion)
- **Attendees:**
- **Action Items Owner:**

## 9. Attachments
- Grafana snapshots
- Logs or alerts
- CI artifacts

---
*Reference materials:*
- `config/monitoring/rollback_triggers.yaml`
- `docs/operations/rollback_drills.md`
- `docs/runbooks/embeddings_service_runbook.md`
