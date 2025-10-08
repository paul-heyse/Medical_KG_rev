# Hard Cutover Strategy: Complete Replacement Architecture

## Executive Summary

This proposal implements a **complete replacement** of the existing orchestration architecture, not a gradual migration. The implementation strategy eliminates all legacy code atomically as new components are added, ensuring:

- ✅ **No feature flags or compatibility shims**
- ✅ **Systematic legacy code decommissioning**
- ✅ **Delegation to open-source libraries (Dagster, Haystack, tenacity, pybreaker, aiolimiter)**
- ✅ **Codebase shrinkage (≥30% reduction in orchestration code)**
- ✅ **Single feature branch with complete replacement**
- ✅ **Rollback = revert entire branch**

---

## Why Hard Cutover (Not Gradual Migration)?

### Current State

The system **is not yet in production**, which provides a unique opportunity for clean replacement without the complexity of maintaining dual orchestration paths.

### Key Advantages

1. **Simplicity**: No feature flags, no A/B testing, no compatibility shims
2. **Code Quality**: Forces complete legacy elimination, prevents code rot
3. **Velocity**: Faster implementation (no dual-path testing, no gradual rollout)
4. **Maintainability**: Cleaner commit history, easier code review
5. **Library Delegation**: Explicit replacement of bespoke code with proven open-source libraries

### Risk Mitigation

- **Comprehensive Testing**: End-to-end tests validate new architecture before merge
- **Performance Benchmarks**: Automated validation of throughput and latency requirements
- **Atomic Commits**: Each commit adds new code + deletes legacy in same transaction
- **Rollback Plan**: Revert entire feature branch if critical issues within 48 hours

---

## Legacy Code Decommissioning Plan

### Phase 1: Audit (Day 1-2)

**Objective**: Identify all code to be deleted and all functionality to delegate to libraries

#### Files to Delete

**Orchestration Core** (`src/Medical_KG_rev/orchestration/`):

- `orchestrator.py` (176 lines) → Replace with Dagster jobs
- `worker.py` (110 lines) → Replace with Dagster ops
- `pipeline.py` → Replace with YAML topology configs
- `profiles.py` → Replace with per-pipeline YAML configs

**Service Layer Bespoke Logic** (`src/Medical_KG_rev/services/`):

- `services/retrieval/chunking.py` (custom splitters) → Replace with HaystackChunker
- `services/embedding/service.py` (bespoke retry logic) → Replace with tenacity decorators
- `services/retrieval/indexing_service.py` → Replace with HaystackIndexWriter

**Custom Resilience** (scattered across codebase):

- All `for attempt in range(max_retries)` loops → tenacity decorators
- All custom failure counting logic → pybreaker decorators
- All `time.sleep()` rate limiting → aiolimiter decorators

#### Dependency Mapping

**Command**: `grep -r "from.*orchestrator import\|Orchestrator\(\\|execute_pipeline" src/`

**Expected Results**:

- Gateway services: 3 imports
- Test fixtures: 8 imports
- Documentation examples: 5 references

**Replacement Plan**: Update all to `from .dagster import DagsterOrchestrator, submit_to_dagster`

### Phase 2: Delegation Validation (Day 3-5)

**Objective**: Verify all bespoke logic is properly delegated to open-source libraries

#### Chunking Delegation to Haystack

**Audit**:

```bash
# Find custom chunking implementations
grep -r "class.*Splitter\|def.*split.*chunk" src/Medical_KG_rev/services/retrieval/
```

**Decision**:

- **Keep**: Profile detection (`detect_profile()` method)
- **Delegate**: All splitting logic to `Haystack DocumentSplitter`
- **Delete**: Custom `SemanticSplitter`, `SlidingWindow` implementations

**Validation**:

```python
# Before: Custom implementation
def chunk_document(doc, profile):
    if profile == "pmc":
        return SemanticSplitter(tau_coh=0.82).split(doc)
    else:
        return SlidingWindow(size=512).split(doc)

# After: Haystack delegation
def chunk_document(doc, profile):
    haystack_splitter = get_haystack_splitter(profile)
    return haystack_splitter.run(documents=[doc])
```

#### Embedding Delegation to Haystack

**Audit**:

```bash
# Find custom embedding loops
grep -r "for.*in.*chunks\|batch_embed\|embedding_loop" src/Medical_KG_rev/services/embedding/
```

**Decision**:

- **Keep**: Namespace management, GPU fail-fast detection
- **Delegate**: All embedding calls to `Haystack OpenAIDocumentEmbedder`
- **Delete**: Custom batch processing logic (Haystack handles batching)

**Validation**:

```python
# Before: Custom batch processing
async def embed_chunks(chunks):
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = await model.encode(batch)
        results.extend(embeddings)
    return results

# After: Haystack delegation
async def embed_chunks(chunks):
    embedder = HaystackEmbedder(namespace="single_vector.bge_small_en.384.v1")
    return embedder.run(documents=chunks)
```

#### Retry Logic Delegation to Tenacity

**Audit**:

```bash
# Find custom retry patterns
grep -r "for attempt in range\|retry_count\|max_retries" src/
```

**Decision**:

- **Delete**: All `for attempt in range(max_retries)` loops
- **Replace**: With `@retry_on_failure` decorator from resilience policies

**Validation**:

```python
# Before: Custom retry loop
async def fetch_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await http_get(url)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

# After: Tenacity decorator
@retry_on_failure(policy="external_api")
async def fetch_with_retry(url):
    return await http_get(url)  # tenacity handles retries
```

#### Circuit Breaker Delegation to pybreaker

**Audit**:

```bash
# Find custom circuit breaker logic
grep -r "failure_count\|circuit.*open\|consecutive_failures" src/
```

**Decision**:

- **Delete**: All bespoke circuit breaker implementations
- **Replace**: With `@circuit_breaker` decorator from resilience policies

**Validation**:

```python
# Before: Custom circuit breaker
class ServiceClient:
    def __init__(self):
        self.failure_count = 0
        self.circuit_open = False

    async def call(self):
        if self.circuit_open:
            raise CircuitOpenError()
        try:
            result = await service_call()
            self.failure_count = 0
            return result
        except Exception:
            self.failure_count += 1
            if self.failure_count >= 5:
                self.circuit_open = True
            raise

# After: pybreaker decorator
@circuit_breaker(policy="external_api")
async def call_service():
    return await service_call()  # pybreaker handles circuit state
```

#### Rate Limiting Delegation to aiolimiter

**Audit**:

```bash
# Find custom rate limiting
grep -r "time\\.sleep\|rate.*limit\|throttle" src/
```

**Decision**:

- **Delete**: All custom rate limiting logic
- **Replace**: With `@rate_limit` decorator from resilience policies

**Validation**:

```python
# Before: Custom rate limiting
class RateLimiter:
    def __init__(self, rate):
        self.rate = rate
        self.last_call = 0

    async def acquire(self):
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < 1.0 / self.rate:
            await asyncio.sleep((1.0 / self.rate) - elapsed)
        self.last_call = time.time()

# After: aiolimiter decorator
@rate_limit(policy="clinicaltrials")
async def fetch_study(nct_id):
    return await api_call(nct_id)  # aiolimiter enforces rate
```

### Phase 3: Atomic Deletion (Day 6-10)

**Objective**: Delete legacy code in same commits as new implementations

#### Commit Strategy

```bash
# Commit 1: Add Dagster jobs + delete orchestrator/worker
git add src/Medical_KG_rev/orchestration/dagster/
git rm src/Medical_KG_rev/orchestration/orchestrator.py
git rm src/Medical_KG_rev/orchestration/worker.py
git commit -m "feat: Add Dagster jobs, delete legacy orchestrator

- Add auto_pipeline_job, pdf_two_phase_job with typed stage contracts
- Add pdf_ir_ready_sensor for two-phase gate
- Delete Orchestrator class (176 lines)
- Delete IngestWorker, MappingWorker (110 lines)
- Total: +450 lines, -286 lines (net +164 for better structure)"

# Commit 2: Add HaystackChunker + delete custom chunkers
git add src/Medical_KG_rev/orchestration/haystack/chunker.py
git rm src/Medical_KG_rev/services/retrieval/chunking.py
git commit -m "feat: Delegate chunking to Haystack DocumentSplitter

- Add HaystackChunker wrapper with profile detection
- Delete custom SemanticSplitter, SlidingWindow implementations
- Total: +120 lines, -240 lines (net -120, delegation achieved)"

# Commit 3: Add HaystackEmbedder + delete custom embedding loops
git add src/Medical_KG_rev/orchestration/haystack/embedder.py
git rm src/Medical_KG_rev/services/embedding/service.py
git commit -m "feat: Delegate embedding to Haystack OpenAIDocumentEmbedder

- Add HaystackEmbedder with namespace management
- Delete custom batch processing, retry logic
- Preserve GPU fail-fast behavior
- Total: +80 lines, -150 lines (net -70, delegation achieved)"

# Commit 4: Add resilience decorators + delete custom retry/circuit breaker
git add src/Medical_KG_rev/orchestration/resilience/
git commit -m "feat: Replace bespoke resilience with tenacity/pybreaker/aiolimiter

- Add @retry_on_failure, @circuit_breaker, @rate_limit decorators
- Load policies from config/orchestration/resilience.yaml
- Delete all custom retry loops (15 occurrences)
- Delete all custom circuit breaker logic (5 implementations)
- Delete all custom rate limiting (8 occurrences)
- Total: +150 lines, -380 lines (net -230, major delegation win)"

# Commit 5: Add HaystackIndexWriter + delete indexing_service
git add src/Medical_KG_rev/orchestration/haystack/index_writer.py
git rm src/Medical_KG_rev/services/retrieval/indexing_service.py
git commit -m "feat: Delegate indexing to Haystack DocumentWriter

- Add HaystackIndexWriter for OpenSearch + FAISS
- Delete custom indexing logic
- Total: +60 lines, -100 lines (net -40)"

# Commit 6: Update all imports
git add src/Medical_KG_rev/orchestration/__init__.py
git add src/Medical_KG_rev/gateway/services.py
git commit -m "refactor: Update imports to use new Dagster orchestration

- Remove Orchestrator, IngestWorker, MappingWorker imports
- Add DagsterOrchestrator, submit_to_dagster
- Update 11 files with import changes"

# Commit 7: Delete legacy tests, add Dagster tests
git rm tests/orchestration/test_orchestrator.py
git rm tests/orchestration/test_workers.py
git rm tests/orchestration/test_integration.py
git add tests/orchestration/test_dagster_jobs.py
git add tests/orchestration/test_dagster_sensors.py
git add tests/orchestration/test_stage_contracts.py
git commit -m "test: Replace legacy orchestration tests with Dagster tests

- Delete 3 legacy test files (480 lines)
- Add 3 new Dagster test files (520 lines)
- Verify stage contract compliance
- Test PDF two-phase gate behavior
- Coverage: 92% (up from 88%)"
```

#### Test After Each Commit

```bash
# Run full test suite
pytest tests/ -v

# Verify no regressions
pytest tests/orchestration/ -v --cov=src/Medical_KG_rev/orchestration

# Check for unused imports
ruff check --select F401 src/

# Type checking
mypy src/Medical_KG_rev/orchestration/
```

### Phase 4: Codebase Validation (Day 11-12)

**Objective**: Verify codebase shrinkage and complete legacy elimination

#### Measurement

```bash
# Before measurements
cloc src/Medical_KG_rev/orchestration/ src/Medical_KG_rev/services/ > BEFORE_CLOC.txt

# After measurements
cloc src/Medical_KG_rev/orchestration/ src/Medical_KG_rev/services/ > AFTER_CLOC.txt

# Calculate reduction
python scripts/calculate_code_reduction.py BEFORE_CLOC.txt AFTER_CLOC.txt
```

**Expected Results**:

- **Before**: ~2,200 lines of Python code
- **After**: ~1,400 lines of Python code
- **Reduction**: 800 lines (36% reduction) ✅ Exceeds 30% target

#### Verification Checklist

- [ ] No references to `Orchestrator` class remain
- [ ] No references to `IngestWorker` or `MappingWorker` remain
- [ ] No `execute_pipeline` calls remain
- [ ] All retry loops use `@retry_on_failure`
- [ ] All circuit breakers use `@circuit_breaker`
- [ ] All rate limiting uses `@rate_limit`
- [ ] All chunking uses `HaystackChunker`
- [ ] All embedding uses `HaystackEmbedder`
- [ ] All indexing uses `HaystackIndexWriter`
- [ ] Test coverage ≥90% for new code
- [ ] No failing tests

---

## Rollback Strategy

### Rollback Trigger Conditions

Execute rollback if any of the following occur **within 48 hours of production deployment**:

1. **Error Rate**: >5% of jobs failing
2. **Performance**: P95 latency >2x baseline or throughput <50% baseline
3. **Data Loss**: Any indication of missing/corrupted documents
4. **GPU Failure**: GPU services unable to start or repeatedly crash
5. **Critical Bug**: Security vulnerability or data corruption bug

### Rollback Procedure

```bash
# Step 1: Identify feature branch commit
FEATURE_COMMIT=$(git log --oneline --grep="feat: Add Dagster jobs" -n 1 | awk '{print $1}')

# Step 2: Create rollback branch
git checkout -b rollback-dagster-orchestration main

# Step 3: Revert feature branch (single revert for entire branch)
git revert -m 1 $FEATURE_COMMIT

# Step 4: Test rollback branch locally
pytest tests/ -v

# Step 5: Deploy rollback to production
git push origin rollback-dagster-orchestration

# Step 6: Update runbook with incident details
echo "Rollback performed at $(date): $REASON" >> docs/runbooks/ROLLBACK_HISTORY.md

# Step 7: Post-mortem
# Schedule retrospective within 24 hours
# Document failure mode, root cause, prevention plan
```

### Post-Rollback Actions

1. **Preserve Logs**: Export all CloudEvents, Prometheus metrics, Dagster UI logs
2. **Root Cause Analysis**: Identify why new architecture failed
3. **Fix Forward Plan**: Create remediation plan for issues
4. **Retry Timeline**: Determine when to attempt re-deployment

---

## Success Criteria

### Code Quality Metrics

- [ ] **Codebase Reduction**: ≥30% fewer lines in orchestration code
- [ ] **Test Coverage**: ≥90% for all new Dagster/Haystack code
- [ ] **No Legacy References**: 0 imports of deleted classes
- [ ] **Lint Clean**: 0 ruff/mypy errors

### Performance Metrics

- [ ] **Throughput**: ≥100 documents/second (baseline: 80 docs/sec)
- [ ] **P95 Latency**: <500ms for retrieval (baseline: 450ms)
- [ ] **P99 Latency**: <2000ms for end-to-end ingestion (baseline: 1800ms)
- [ ] **GPU Utilization**: ≥80% (no wasted GPU cycles)

### Operational Metrics

- [ ] **Dagster UI Accessible**: <2s load time, 99.9% uptime
- [ ] **Sensor Reliability**: PDF gate sensor triggers within 30s of ledger update
- [ ] **CloudEvents Stream**: 100% of stage lifecycle events emitted
- [ ] **Error Rate**: <2% job failure rate (baseline: 1.5%)

### Delegation Completeness

- [ ] **Chunking**: 100% delegated to Haystack
- [ ] **Embedding**: 100% delegated to Haystack
- [ ] **Indexing**: 100% delegated to Haystack
- [ ] **Retries**: 100% use tenacity decorators
- [ ] **Circuit Breakers**: 100% use pybreaker
- [ ] **Rate Limiting**: 100% use aiolimiter

---

## Communication Plan

### Stakeholder Notification

**Before Implementation** (Day 0):

- Email to engineering team: "DAG-based orchestration implementation starting Monday"
- Slack notification: Pin message with link to proposal
- Standups: Brief overview of hard cutover strategy

**During Implementation** (Day 1-12):

- Daily standups: Progress update, blockers
- Commit notifications: Automated Slack messages for each atomic deletion commit
- PR reviews: At least 2 reviewers for each commit

**Before Deployment** (Day 13):

- Email to engineering + operations: "Production deployment scheduled for [date]"
- Runbook review: Operations team walkthrough of new Dagster UI
- On-call briefing: Emergency rollback procedure

**After Deployment** (Day 14-16):

- Monitor dashboard: Grafana dashboard shared in Slack
- Status updates: Every 6 hours for first 48 hours
- Retrospective: Scheduled within 1 week

---

## Documentation Updates

### Files to Update

**Remove Legacy References**:

- `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`: Remove Section 6.2 "Legacy Pipeline Stages"
- `docs/guides/orchestration-pipelines.md`: Remove hardcoded pipeline examples
- `README.md`: Remove "Start background workers" command

**Add New Documentation**:

- `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`: Add Section 6.2 "Dagster Job Definitions"
- `docs/guides/orchestration-pipelines.md`: Add YAML topology examples
- `README.md`: Add "Start Dagster daemon" command
- `DELETED_CODE.md`: Document what was removed and why

### Code Reduction Report Template

```markdown
# Codebase Reduction Report: Dagster Orchestration

## Summary

- **Date**: [YYYY-MM-DD]
- **Feature Branch**: add-dag-orchestration-pipeline
- **Lines Removed**: 956 lines
- **Lines Added**: 600 lines
- **Net Reduction**: 356 lines (36% reduction)

## Deleted Files

| File | Lines | Replacement |
|------|-------|-------------|
| `orchestrator.py` | 176 | Dagster jobs |
| `worker.py` | 110 | Dagster ops |
| `chunking.py` | 240 | HaystackChunker |
| `embedding/service.py` | 150 | HaystackEmbedder |
| `indexing_service.py` | 100 | HaystackIndexWriter |
| Custom retry logic | 180 | tenacity decorators |

## Delegation Achieved

| Category | Before (LOC) | After (LOC) | Library |
|----------|--------------|-------------|---------|
| Chunking | 240 | 120 | Haystack DocumentSplitter |
| Embedding | 150 | 80 | Haystack OpenAIDocumentEmbedder |
| Indexing | 100 | 60 | Haystack DocumentWriter |
| Retries | 180 | 50 | tenacity |
| Circuit Breakers | 120 | 40 | pybreaker |
| Rate Limiting | 80 | 30 | aiolimiter |

## Validation

- ✅ Test coverage: 92% (up from 88%)
- ✅ No legacy imports remain
- ✅ All tests passing
- ✅ Performance: Throughput +15%, P95 latency -8%
```

---

## Conclusion

This hard cutover strategy ensures:

1. **Clean Replacement**: No legacy code remains in codebase
2. **Library Delegation**: All bespoke logic replaced with proven open-source libraries
3. **Code Quality**: Codebase shrinks by 36%, test coverage increases to 92%
4. **Operational Safety**: Comprehensive testing, clear rollback procedure, 48-hour monitoring window
5. **Team Alignment**: Clear communication plan, atomic commits, thorough documentation

**Status**: Ready for implementation

**Next Steps**:

1. Review this strategy document with team
2. Create feature branch: `git checkout -b add-dag-orchestration-pipeline main`
3. Execute Phase 1: Audit and create `LEGACY_DECOMMISSION_CHECKLIST.md`
4. Begin atomic deletions per commit strategy
