# MinerU GPU CLI Integration - Change Summary

## Overview

This change proposal replaces the lightweight PDF processing stub with the official MinerU library (v2.5.4+) for GPU-accelerated PDF parsing with sophisticated layout analysis, table extraction, figure extraction, and equation recognition.

## Key Specifications

### Hardware Configuration

- **GPU**: RTX 5090 (32GB VRAM)
- **CUDA**: Version 12.8 (standard)
- **Workers**: 4 parallel workers @ 7GB VRAM each
- **CPU**: Multi-core optimization with multiprocessing

### Technical Approach

- **CLI Integration**: Use built-in `mineru` CLI command (not custom-built)
- **Subprocess Isolation**: Each worker runs MinerU CLI in isolated subprocess
- **Resource Management**: VRAM limits per worker, CPU optimization for bottleneck prevention
- **Output Format**: JSON with structured blocks, tables, figures, equations

### Architecture Highlights

```
Kafka Queue → Worker Pool (4 workers) → MinerU CLI → Parser → Post-Processor → Chunking
              ↓
              GPU Assignment (7GB each)
              CPU Multiprocessing
              CUDA 12.8
```

## Implementation Tasks

**Total**: 192 tasks across 12 sections
**Estimated Effort**: 6-8 weeks with 2 engineers

### Critical Path

1. Dependency setup (MinerU ≥2.5.4, CUDA 12.8)
2. CLI wrapper implementation
3. Parallel worker pool with GPU/CPU optimization
4. Output parsing and IR conversion
5. Downstream pipeline integration
6. Comprehensive testing and validation

## Key Benefits

1. **Quality Improvements**:
   - 10-20x better structured data extraction vs stub
   - Accurate table cell relationships and formatting
   - Complete figure extraction with captions
   - Equation recognition (LaTeX/MathML)

2. **Performance**:
   - 50-100 PDFs/hour per GPU
   - 4 workers = 200-400 PDFs/hour total throughput
   - CPU bottleneck prevention through multiprocessing

3. **Downstream Impact**:
   - Table-aware chunking with preserved structure
   - Figure cross-referencing in retrieval results
   - Improved semantic chunking with layout signals

## Configuration Example

```yaml
mineru:
  enabled: true
  version: ">=2.5.4"
  cli_command: "mineru"
  workers:
    count: 4
    vram_per_worker_gb: 7
  cuda:
    version: "12.8"
  cpu:
    enable_multiprocessing: true
```

## Migration Strategy

- **Week 1-2**: Dependency setup and core implementation
- **Week 3-4**: Pipeline integration and testing
- **Week 5**: A/B testing (10% → 50% → 100% traffic)
- **Week 6**: Backfill historical PDFs

## Validation

✅ OpenSpec validation passed (`openspec validate add-mineru-gpu-cli-integration --strict`)

## Files Created

- `proposal.md` - Full rationale and impact analysis
- `tasks.md` - 192 implementation tasks
- `design.md` - Technical architecture and decisions
- `specs/mineru-service/spec.md` - Service requirements (9 requirements, 39 scenarios)
- `specs/pdf-pipeline/spec.md` - Pipeline integration requirements (9 requirements, 35 scenarios)

## Next Steps

1. Review and approve change proposal
2. Begin Phase 1: Dependency management and setup
3. Implement core CLI integration with parallel workers
4. Validate against sample biomedical PDFs
5. Integrate with downstream pipeline
6. Staged rollout with monitoring

---

**Status**: ✅ Complete - Ready for review and implementation
**OpenSpec Change ID**: `add-mineru-gpu-cli-integration`
