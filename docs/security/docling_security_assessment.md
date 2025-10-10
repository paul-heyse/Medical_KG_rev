# Docling VLM Security Assessment

## Overview
Docling VLM replaces the legacy MinerU OCR stack with a Gemma3 12B vision-language model that generates structured document understanding artifacts. This assessment summarizes the security posture of the new processing flow and the mitigations required for production adoption.

## Dependency Review
- **docling[vlm] 2.x** – Provides Docling orchestration, tokenizer utilities, and runtime services. Packages pull transitive dependencies such as `torch`, `transformers`, and `sentencepiece`. All dependencies are reviewed quarterly for CVEs via Dependabot and the internal SBOM scanner.
- **Gemma3 12B checkpoints** – Distributed through Hugging Face with SHA256 verification. Artifacts are stored in an encrypted object store with KMS managed keys.
- **torch 2.1+ (CUDA 12.1)** – Evaluated for GPU side-channel and memory safety advisories. Runtime containers pin the exact CUDA minor version to avoid drift.
- **transformers 4.36+** – Subject to static analysis for deserialization vulnerabilities. Only vetted model architectures are enabled (Gemma family); arbitrary `AutoModel` loading is disabled in configuration.

## Threat Model
| Threat | Risk | Mitigation |
| --- | --- | --- |
| GPU memory residue leaks PHI between jobs | Medium | Enforce `torch.cuda.empty_cache()` after each batch, scrub intermediate tensors, and allocate dedicated MIG slices for Docling workloads. |
| Malicious PDF triggers remote code execution via model tokenizer | Low | All PDFs run through the existing sanitization/validation pipeline prior to VLM processing; `transformers` deserialization is restricted to trusted configs. |
| Compromised Gemma3 weights | Medium | Signatures verified on download, hashes re-validated nightly, and provenance tracked in ledger entries. |
| Over-permissioned service accounts expose Docling config | Medium | Docling feature flag and model paths stored in Secrets Manager with least-privilege IAM roles; RBAC policies updated to remove MinerU permissions. |
| Network exfiltration from GPU nodes | Low | GPU worker nodes remain in private subnets with egress firewall rules limiting outbound traffic to artifact repositories. |

## Mitigations & Controls
1. **Runtime Hardening**
   - Enable NVIDIA container toolkit sandboxing, disable host PID/IPC sharing, and mount model cache read-only.
   - Enforce AppArmor profile `docling-vlm` restricting filesystem access to `/models/gemma3-12b` and `/tmp/docling`.
2. **Observability**
   - Prometheus metrics emit Docling success/failure counts and GPU memory levels. Alerting thresholds configured in `config/monitoring/alerts.yml` trigger PagerDuty notifications.
   - Structured audit logs capture operator ID, correlation ID, and processing config for every request.
3. **Data Protection**
   - PDFs and generated artifacts encrypted with AES-256 at rest. In-transit encryption uses mTLS between gateway and Docling workers.
   - Temporary files removed within 60 seconds of job completion via secure deletion routine.
4. **Access Controls**
   - New `docling:process` and `docling:admin` scopes restrict API usage. Admin actions require MFA with hardware security keys.
   - CI secrets rotated to remove MinerU credentials and provision Docling model tokens.
5. **Incident Response**
   - Rollback script `scripts/rollback_to_mineru.sh` preserves archived MinerU stack for emergency failover but requires security approval before activation.
   - Runbook updated with Docling-focused containment steps, including model cache revocation and token disable procedures.

## Residual Risk
With the mitigations above, residual risk is assessed as **Low**. Remaining exposure centers on third-party model dependencies and GPU isolation guarantees. Continuous monitoring plus quarterly security reviews are mandated to keep the Docling stack compliant with HIPAA and internal security baselines.
