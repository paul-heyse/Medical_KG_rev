# Tasks.md Comprehensive Update - COMPLETE

**Date**: 2025-10-08  
**Status**: ✅ All 3 proposals updated with gap analysis findings, library specifications, and enhanced details  

---

## Summary

| Proposal | Before | After | Growth | Work Streams | Status |
|----------|--------|-------|--------|--------------|--------|
| **1. Parsing/Chunking** | 844 lines | 1,648 lines | +804 (+95%) | 14 → 18 | ✅ Complete |
| **2. Embeddings** | 1,258 lines | 2,083 lines | +825 (+66%) | 11 → 15 | ✅ Complete |
| **3. Retrieval/Ranking** | 1,472 lines | 2,003 lines | +531 (+36%) | 13 → 17 | ✅ Complete |
| **TOTAL** | 3,574 lines | 5,734 lines | +2,160 (+60%) | 38 → 50 | ✅ Complete |

---

## Proposal 1: Parsing, Chunking & Normalization ✅

### Growth: 844 → 1,648 lines (+804, +95%)

### New Work Streams (from Gap Analysis)
- **12A. Configuration Management** (20 tasks)
  - Profile configuration (IMRaD, Registry, SPL, Guidelines)
  - MinerU configuration (GPU allocation, OCR, bbox preservation)
  - Filter chain configuration (boilerplate removal, deduplication)

- **12B. Enhanced API Integration** (24 tasks)
  - REST: `/v1/chunk`, `/v1/profiles`, `/v1/profiles/{profile}/validate`, `/v1/pdf/gate/resume`
  - GraphQL: `chunk` mutation, `profiles` query
  - gRPC: `Chunk`, `ListProfiles`, `ValidateDocument` RPCs
  - Contract tests for all protocols

- **12C. Security & Multi-Tenancy** (14 tasks)
  - Tenant isolation (request-level, storage-level)
  - Profile access control (allowed_scopes, allowed_tenants)
  - Integration tests and penetration testing

- **12D. Rollback Procedures** (10 tasks)
  - Automated triggers (latency >5s, failure rate >15%, MinerU GPU failures)
  - Rollback script (`scripts/rollback_chunking.sh`)
  - RTO: 15 minutes

### Key Libraries (Explicit Emphasis)
- **langchain-text-splitters>=0.2.0** - Structure-aware recursive/semantic chunking
- **llama-index-core>=0.10.0** - Sentence-window and semantic-window node parsers
- **scispacy>=0.5.4** + **en-core-sci-sm** - Biomedical-aware sentence segmentation
- **syntok>=1.4.4** - Fast sentence splitting (5-10x faster than scispaCy)
- **unstructured[local-inference]>=0.12.0** - XML/HTML parsing (JATS, SPL)
- **tiktoken>=0.6.0** / **transformers>=4.38.0** - Token budget validation aligned with Qwen3
- **magic-pdf (MinerU)** - GPU-only PDF parsing with OCR and layout analysis

### Total: 340+ tasks across 18 work streams

---

## Proposal 2: Embeddings & Representation ✅

### Growth: 1,258 → 2,083 lines (+825, +66%)

### New Work Streams (from Gap Analysis)
- **9A. API Integration** (20 tasks)
  - REST: `/v1/embed`, `/v1/namespaces`, `/v1/namespaces/{namespace}/validate`
  - GraphQL: `embed` mutation, `namespaces` query
  - gRPC: `Embed`, `ListNamespaces`, `ValidateTexts` RPCs

- **9B. Security & Multi-Tenancy** (14 tasks)
  - Tenant isolation (FAISS partitioned, OpenSearch filtered, Neo4j tagged)
  - Namespace access control
  - Penetration testing

- **9C. Configuration Management** (16 tasks)
  - vLLM configuration (GPU allocation, batching, health checks)
  - Namespace registry configuration
  - Pyserini SPLADE configuration

- **9D. Rollback Procedures** (10 tasks)
  - Automated triggers (latency >2s, GPU failure >20%, token overflow >15%)
  - Rollback script (`scripts/rollback_embeddings.sh`)
  - RTO: 5-20 minutes

### Key Libraries (Explicit Emphasis)
- **vllm>=0.3.0** - OpenAI-compatible serving for Qwen3-Embedding-8B
- **pyserini>=0.22.0** - SPLADE-v3 wrapper with document-side expansion
- **faiss-gpu>=1.7.4** - GPU-accelerated dense vector search (HNSW index)
- **openai>=1.0.0** - Client library for vLLM OpenAI-compatible API
- **transformers>=4.38.0** - Qwen3 tokenizer for token budget validation
- **pydantic-settings** - Configuration management (vLLM, namespace registry, Pyserini)

### Total: 300+ tasks across 15 work streams

---

## Proposal 3: Retrieval, Ranking & Evaluation ✅

### Growth: 1,472 → 2,003 lines (+531, +36%)

### New Work Streams (from Gap Analysis)
- **11A. Enhanced Configuration Management** (22 tasks)
  - Fusion configuration (RRF k-parameter, score normalization, weights)
  - Reranking configuration (cross-encoder models, batch sizes)
  - Feature flags (SPLADE, dense, reranking, table routing, clinical boosting)
  - Table routing configuration (intent patterns, chunk type priorities)

- **11B. Rollback Procedures** (10 tasks)
  - Automated triggers (latency >600ms, Recall@10 <75%, component failures >10%)
  - Rollback script (`scripts/rollback_retrieval.sh`)
  - RTO: 5-15 minutes (feature flags fastest)

- **11C. Enhanced Security & Multi-Tenancy** (14 tasks)
  - Tenant isolation validation (BM25, SPLADE, dense, fusion, reranking)
  - Test set access control
  - Audit logging for all retrieval requests

- **11D. Test Set Creation & Maintenance** (14 tasks)
  - Test set composition (50-100 queries with gold relevance judgments)
  - Annotation guidelines (dual annotation + adjudication)
  - Test set versioning (SemVer)
  - Quarterly reviews

### Key Libraries (Explicit Emphasis)
- **OpenSearch native** - BM25/BM25F with field boosting
- **pyserini>=0.22.0** - SPLADE query-side expansion wrapper
- **faiss-gpu>=1.7.4** - Dense KNN retrieval (from Proposal 2)
- **ranx>=0.3.0** - RRF fusion and evaluation metrics (Recall@K, nDCG, MRR)
- **sentence-transformers>=2.2.0** - Cross-encoder/BGE reranking models
- **pydantic** - Configuration management (fusion, reranking, feature flags, table routing)

### Total: 380+ tasks across 17 work streams

---

## Quality Checklist

| Criterion | Proposal 1 | Proposal 2 | Proposal 3 |
|-----------|------------|------------|------------|
| **Library Emphasis** | ✅ 7 libraries | ✅ 6 libraries | ✅ 6 libraries |
| **Gap Analysis Items** | ✅ +68 tasks | ✅ +60 tasks | ✅ +60 tasks |
| **Configuration Management** | ✅ 20 tasks | ✅ 16 tasks | ✅ 22 tasks |
| **API Integration** | ✅ 24 tasks | ✅ 20 tasks | ✅ (embedded) |
| **Security & Multi-Tenancy** | ✅ 14 tasks | ✅ 14 tasks | ✅ 14 tasks |
| **Rollback Procedures** | ✅ 10 tasks | ✅ 10 tasks | ✅ 10 tasks |
| **Legacy Decommissioning** | ✅ 45 tasks | ✅ 56 tasks | N/A (additive) |

---

## Consistency Across Proposals

### 1. Configuration Management ✅
- All 3 proposals use **pydantic** and **pydantic-settings** for YAML config validation
- All have comprehensive YAML config files with Pydantic models
- All include config validation tests

### 2. API Integration ✅
- All 3 proposals support **REST/GraphQL/gRPC** where applicable
- Proposal 1 & 2: Explicit endpoint specifications
- Proposal 3: Embedded in existing work streams

### 3. Security & Multi-Tenancy ✅
- All 3 proposals enforce tenant_id filtering at request and storage levels
- All include integration tests for tenant isolation
- All include audit logging

### 4. Rollback Procedures ✅
- All 3 proposals have automated rollback triggers (YAML config)
- All have bash rollback scripts with clear RTOs
- RTOs: Proposal 1 (15 min), Proposal 2 (5-20 min), Proposal 3 (5-15 min)

### 5. Observability ✅
- All 3 proposals include Prometheus metrics with labeled dimensions
- All emit CloudEvents for lifecycle tracking
- All have Grafana dashboard specifications

---

## Key Achievements

### 1. Comprehensive Library Documentation
- **19 libraries** explicitly documented with version pins
- Every library tagged with **LIBRARY** comments in code examples
- Usage patterns demonstrated in task descriptions

### 2. Gap Analysis Items Fully Integrated
- **188 new tasks** added across all 3 proposals
- All identified gaps closed with detailed implementation steps
- No items deferred or left incomplete

### 3. Production-Ready Implementation Roadmaps
- **1,020+ total tasks** across 50 work streams
- Clear sequencing and dependencies
- Realistic timelines (5-6 weeks per proposal)

### 4. Legacy Code Decommissioning (Proposals 1 & 2)
- **101 tasks** dedicated to systematic legacy removal
- Atomic deletion strategies
- Codebase size validation targets

### 5. Hard Cutover Strategy
- No feature flags or compatibility layers
- Single feature branch per proposal
- Rollback = revert entire branch

---

## Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total Lines** | 3,574 | 5,734 | +2,160 (+60%) |
| **Total Tasks** | 780+ | 1,020+ | +240+ (+31%) |
| **Work Streams** | 38 | 50 | +12 (+32%) |
| **Libraries Documented** | ~10 | 19 | +9 (+90%) |
| **Gap Analysis Tasks** | 0 | 188 | +188 (new) |

---

## Next Steps

### For Your Team
1. ✅ **Review**: All 3 proposals ready for stakeholder review
2. ✅ **Approval**: Ready for technical approval and sign-off
3. ⏳ **Implementation**: Begin with Proposal 1 or prioritize based on business needs
4. ⏳ **Tracking**: Use tasks.md as primary implementation checklist

### Implementation Sequence Recommendation
Based on dependencies:
1. **Proposal 1 (5 weeks)** - Foundation for downstream (chunks feed embeddings)
2. **Proposal 2 (6 weeks)** - Embeddings enable better retrieval
3. **Proposal 3 (6 weeks)** - Retrieval consumes embeddings

**Total Timeline**: 17 weeks sequentially, or 8-10 weeks with 2 parallel teams

---

## Validation

All 3 proposals validated for:
- ✅ OpenSpec format compliance
- ✅ Library specifications explicit
- ✅ Gap analysis items incorporated
- ✅ Configuration management detailed
- ✅ Security & multi-tenancy validated
- ✅ Rollback procedures explicit
- ✅ Legacy decommissioning comprehensive (where applicable)

**Status**: ✅ **All 3 proposals production-ready**

