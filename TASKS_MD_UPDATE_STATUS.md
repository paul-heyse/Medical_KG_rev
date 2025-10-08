# Tasks.md Update Status - All 3 Proposals

**Date**: 2025-10-08  
**Objective**: Ensure all tasks.md files reflect gap analysis findings, library specifications, and legacy decommissioning details  

---

## Proposal 2: Embeddings & Representation ✅ COMPLETE

### Status
**100% COMPLETE** - Comprehensive update with all gap analysis findings incorporated

### Changes Made
- **Before**: 1,258 lines, 11 work streams
- **After**: 2,083 lines (+825, +66%), 15 work streams

### New Work Streams Added (from Gap Analysis)
1. **9A. API Integration** (20 tasks)
   - REST API: `/v1/embed`, `/v1/namespaces`, `/v1/namespaces/{namespace}/validate`
   - GraphQL API: `embed` mutation, `namespaces` query
   - gRPC API: `Embed`, `ListNamespaces`, `ValidateTexts` RPCs
   - Contract tests (Schemathesis, GraphQL Inspector, Buf)

2. **9B. Security & Multi-Tenancy** (14 tasks)
   - Tenant isolation (request-level, storage-level)
   - FAISS indices partitioned by tenant_id
   - OpenSearch tenant_id filtering
   - Neo4j tenant_id tagging
   - Namespace access control (allowed_scopes, allowed_tenants)
   - Penetration testing

3. **9C. Configuration Management** (16 tasks)
   - vLLM configuration (YAML + Pydantic models)
   - Namespace registry configuration
   - Pyserini SPLADE configuration
   - Config validation tests

4. **9D. Rollback Procedures** (10 tasks)
   - Automated rollback triggers (latency, GPU failure, token overflow)
   - Rollback script (`scripts/rollback_embeddings.sh`)
   - RTO targets (5-20 minutes)
   - Post-incident analysis

### Library Emphasis
✅ **Explicitly documented throughout**:
- `vllm>=0.3.0` - OpenAI-compatible serving
- `pyserini>=0.22.0` - SPLADE-v3 wrapper
- `faiss-gpu>=1.7.4` - Dense vector search
- `openai>=1.0.0` - vLLM client
- `transformers>=4.38.0` - Qwen3 tokenizer
- `pydantic-settings` - Configuration management

### Legacy Decommissioning
✅ **Comprehensive** (Work Stream #1, 56 tasks):
- Inventory & dependency analysis
- Delegation validation (vLLM, Pyserini, tokenizers)
- Atomic deletion strategy
- Import cleanup
- Test migration
- Codebase size validation

### Summary Table
| Metric | Value |
|--------|-------|
| Total Tasks | 300+ |
| Work Streams | 15 |
| Gap Analysis Additions | +60 tasks |
| Library References | 6 key libraries with versions |
| Legacy Decommissioning | 56 tasks |
| Breaking Changes | 4 documented |
| Timeline | 6 weeks |

---

## Proposal 1: Parsing, Chunking & Normalization ⚠️ NEEDS UPDATE

### Current Status
- **Lines**: 844
- **Work Streams**: ~14
- **Gap Analysis Integration**: Partial

### Items Identified for Addition

#### 1. Missing: Enhanced Library Specifications
**Current**: Libraries mentioned but not emphasized enough
**Need**: Explicit library usage throughout with version pins

**Key Libraries to Emphasize**:
- `langchain-text-splitters>=0.0.1` - Structure-aware segmentation
- `llama-index>=0.9.0` - Sentence/semantic-window chunking
- `scispacy>=0.5.0` + `syntok>=1.4.0` - Biomedical sentence segmentation
- `transformers>=4.38.0` / `tiktoken>=0.5.0` - Token budgets
- `unstructured>=0.10.0` - HTML/oddball documents
- `magic-pdf` (MinerU) - GPU-only PDF parsing

#### 2. Missing: Detailed Configuration Management
**Need**: Add tasks for:
- ChunkerPort interface configuration (IMRaD, Registry, SPL, Guidelines profiles)
- MinerU configuration (GPU allocation, page/bbox mapping)
- Filter chain configuration (boilerplate removal, deduplication)

#### 3. Missing: Comprehensive API Integration
**Current**: Section 11 (API Integration) exists but lacks detail
**Need**: Expand with:
- REST endpoints for chunking profiles
- GraphQL mutations for profile-aware chunking
- gRPC proto for chunking service

#### 4. Missing: Rollback Procedures
**Current**: Line 840 mentions "Emergency rollback" but not detailed
**Need**: Add comprehensive work stream:
- Automated rollback triggers
- Rollback script
- RTO specifications (similar to Proposal 2)

#### 5. Missing: Security & Multi-Tenancy for Chunking
**Need**: Add tasks for:
- Tenant isolation in chunk storage
- Profile access control
- Audit logging

### Recommended Actions
1. Add Work Stream: "Configuration Management" (15-20 tasks)
2. Expand Work Stream 11: "API Integration" (+10 tasks for detailed endpoints)
3. Add Work Stream: "Rollback Procedures" (10 tasks)
4. Add Work Stream: "Security & Multi-Tenancy" (12 tasks)
5. Update summary to emphasize libraries explicitly

**Estimated Addition**: +400-500 lines

---

## Proposal 3: Retrieval, Ranking & Evaluation ⚠️ NEEDS UPDATE

### Current Status
- **Lines**: 1,472
- **Work Streams**: 13
- **Gap Analysis Integration**: Moderate

### Items Identified for Addition

#### 1. Missing: Enhanced Library Specifications
**Current**: Some libraries mentioned
**Need**: Explicit emphasis throughout

**Key Libraries to Emphasize**:
- **BM25**: `rank-bm25>=0.2.2` or OpenSearch native
- **SPLADE**: `pyserini>=0.22.0` (query-side expansion)
- **Dense KNN**: `faiss-gpu>=1.7.4` for FAISS operations
- **Fusion**: Custom RRF implementation or `ranx>=0.3.0`
- **Reranking**: `sentence-transformers>=2.2.0` for cross-encoder/BGE
- **Evaluation**: `ranx>=0.3.0` for Recall@K, nDCG, MRR

#### 2. Missing: Detailed Configuration Management
**Need**: Add tasks for:
- Fusion configuration (RRF k-parameter, score normalization weights)
- Reranking model configuration (cross-encoder selection)
- Feature flags configuration (SPLADE on/off, reranking on/off)
- Table-aware routing configuration

#### 3. Missing: Comprehensive Rollback Procedures
**Current**: No rollback section
**Need**: Add work stream:
- Automated rollback triggers (latency degradation, quality drop)
- Rollback script
- RTO specifications

#### 4. Missing: Enhanced Security & Multi-Tenancy
**Current**: Work Stream #8 exists but may need expansion
**Need**: Validate completeness:
- Tenant filtering in all retrieval paths (BM25, SPLADE, dense)
- Index partitioning strategies
- Cross-tenant query prevention

#### 5. Missing: Test Set Creation & Maintenance
**Current**: Work Stream #7 (Evaluation) exists
**Need**: Expand with:
- Test set composition (50-100 queries, gold relevance judgments)
- Annotation guidelines
- Test set versioning

### Recommended Actions
1. Add Work Stream: "Configuration Management (Fusion, Reranking, Features)" (15 tasks)
2. Add Work Stream: "Rollback Procedures" (10 tasks)
3. Expand Work Stream #7: "Evaluation" (+10 tasks for test set creation)
4. Validate Work Stream #8: "API Integration" (ensure REST/GraphQL/gRPC complete)
5. Update summary to emphasize libraries explicitly

**Estimated Addition**: +350-450 lines

---

## Overall Status

| Proposal | Current Lines | Target Lines | Status | Est. Work |
|----------|---------------|--------------|--------|-----------|
| **1. Parsing/Chunking** | 844 | ~1,300 | ⚠️ 65% | +450 lines |
| **2. Embeddings** | 2,083 | 2,083 | ✅ 100% | Complete |
| **3. Retrieval/Ranking** | 1,472 | ~1,900 | ⚠️ 75% | +400 lines |

---

## Recommendations

### Immediate Priority
1. **Complete Proposal 1 tasks.md** - Add 4 missing work streams (~450 lines)
2. **Complete Proposal 3 tasks.md** - Add 3 missing work streams (~400 lines)

### Quality Checklist (All Proposals)
- [x] **Proposal 2**: Library usage explicit with version pins ✅
- [ ] **Proposal 1**: Library usage explicit with version pins
- [ ] **Proposal 3**: Library usage explicit with version pins
- [x] **Proposal 2**: Gap analysis items incorporated ✅
- [ ] **Proposal 1**: Gap analysis items incorporated
- [ ] **Proposal 3**: Gap analysis items incorporated
- [x] **Proposal 2**: Legacy decommissioning comprehensive ✅
- [ ] **Proposal 1**: Legacy decommissioning comprehensive (needs validation)
- [ ] **Proposal 3**: Legacy decommissioning comprehensive (needs validation)
- [x] **Proposal 2**: Configuration management detailed ✅
- [ ] **Proposal 1**: Configuration management detailed
- [ ] **Proposal 3**: Configuration management detailed
- [x] **Proposal 2**: Rollback procedures explicit ✅
- [ ] **Proposal 1**: Rollback procedures explicit
- [ ] **Proposal 3**: Rollback procedures explicit
- [x] **Proposal 2**: Security & multi-tenancy validated ✅
- [ ] **Proposal 1**: Security & multi-tenancy validated
- [ ] **Proposal 3**: Security & multi-tenancy validated

---

## Next Steps

1. **User Approval**: Confirm priority (complete Proposal 2 only, or all 3?)
2. **Proposal 1 Update**: Add missing work streams (~2-3 hours)
3. **Proposal 3 Update**: Add missing work streams (~2-3 hours)
4. **Final Validation**: Ensure all 3 proposals at same quality level

---

**Status**: Proposal 2 complete (2,083 lines, 300+ tasks, 15 work streams)  
**Remaining**: Proposals 1 & 3 need additional work streams from gap analysis
