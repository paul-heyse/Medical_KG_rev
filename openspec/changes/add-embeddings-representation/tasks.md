# Implementation Tasks: Standardized Embeddings & Representation

## CRITICAL: Hard Cutover Strategy

**No Legacy Compatibility**: This is a complete replacement of the embedding architecture with NO transition period and NO accommodation for legacy code.

**Key Principles**:

1. **Atomic Deletions**: Legacy code deleted in same commits as new implementations complete
2. **Library Delegation**: All functionality delegated to vLLM, Pyserini, FAISS (no bespoke code)
3. **Codebase Shrinkage**: Validate 25% reduction (530 → 400 lines)
4. **GPU-Only Enforcement**: 100% fail-fast, zero CPU fallbacks

---

## Work Stream #1: Legacy Code Decommissioning Plan

### Phase 1A: Inventory & Dependency Analysis (Week 1, Days 1-2)

#### 1.1.1 Audit Existing Embedding Code

- [ ] **1.1.1a** List all embedding-related files in `src/Medical_KG_rev/services/embedding/`:
  - `bge_embedder.py` (180 lines)
  - `splade_embedder.py` (210 lines)
  - `manual_batching.py` (95 lines)
  - `token_counter.py` (45 lines)
  - `registry.py` (partial, 120 lines to refactor)
  - `__init__.py` (exports)

- [ ] **1.1.1b** Identify all imports of legacy embedding code across codebase:

  ```bash
  rg "from Medical_KG_rev.services.embedding import (BGEEmbedder|SPLADEEmbedder|ManualBatcher)" --type py
  rg "import.*bge_embedder|splade_embedder|manual_batching" --type py
  ```

- [ ] **1.1.1c** Document current embedding call sites (estimated 15-20 locations):
  - Gateway REST endpoints (`gateway/rest/embedding.py`)
  - Orchestration embedding stage (`orchestration/stages/embed.py`)
  - Chunking service (`services/chunking/service.py`)
  - Test files (`tests/services/test_embedding.py`, `tests/integration/test_embedding_pipeline.py`)

- [ ] **1.1.1d** Measure baseline metrics:
  - Lines of code: `cloc src/Medical_KG_rev/services/embedding/`
  - File count
  - Import count
  - Test coverage for legacy code

#### 1.1.2 Dependency Graph Analysis

- [ ] **1.1.2a** Map dependencies of legacy embedding code:
  - Which orchestration stages depend on `BGEEmbedder`?
  - Which API endpoints expose embedding functionality?
  - Which storage layers expect legacy embedding formats?

- [ ] **1.1.2b** Identify circular dependencies or tight coupling:

  ```bash
  pydeps src/Medical_KG_rev/services/embedding/ --show-deps
  ```

- [ ] **1.1.2c** Document external library usage by legacy code:
  - `sentence-transformers` (used by `bge_embedder.py`)
  - `transformers` (used by `splade_embedder.py`)
  - Custom batching logic (used by `manual_batching.py`)

#### 1.1.3 Test Coverage Audit

- [ ] **1.1.3a** List all tests covering legacy embedding code:

  ```bash
  rg "BGEEmbedder|SPLADEEmbedder|ManualBatcher" tests/ --type py
  ```

- [ ] **1.1.3b** Categorize tests:
  - Unit tests (mock-heavy, test specific embedder logic)
  - Integration tests (test embedding + storage pipeline)
  - Contract tests (test API schema compliance)

- [ ] **1.1.3c** Identify tests to migrate vs delete:
  - Tests of embedder internals → DELETE (vLLM/Pyserini handle this)
  - Tests of embedding API contracts → MIGRATE (rewrite for vLLM client)
  - Tests of storage integration → MIGRATE (update for FAISS/OpenSearch)

---

### Phase 1B: Delegation Validation to New Libraries (Week 1, Days 3-5)

#### 1.2.1 Dense Embedding Delegation (vLLM)

**Goal**: Prove vLLM OpenAI-compatible API replaces all `BGEEmbedder` functionality

- [ ] **1.2.1a** Map `BGEEmbedder` methods to vLLM endpoints:
  - `embed(texts: list[str]) -> np.ndarray` → `POST /v1/embeddings` with `input=[...]`
  - `embed_query(text: str) -> np.ndarray` → `POST /v1/embeddings` with `input="..."`
  - Batching logic → vLLM handles batching internally

- [ ] **1.2.1b** Validate vLLM covers edge cases:
  - Empty text → vLLM returns error (acceptable)
  - Text exceeding token limit → vLLM returns error (acceptable, aligns with fail-fast)
  - Unicode handling → vLLM tokenizer handles NFKC normalization

- [ ] **1.2.1c** Performance parity validation:
  - Benchmark: `BGEEmbedder` throughput vs vLLM throughput
  - Target: vLLM ≥5x faster (100-200 emb/sec → 1000+ emb/sec)
  - GPU memory: vLLM should use ≤16GB for Qwen3-Embedding-8B

- [ ] **1.2.1d** Document delegation:
  - Create table: Legacy Method → vLLM Endpoint → Notes
  - Example:

    | Legacy | vLLM | Notes |
    |--------|------|-------|
    | `BGEEmbedder.embed(texts)` | `POST /v1/embeddings` | vLLM batches internally |
    | `BGEEmbedder.encode(text)` | Same as above | No separate "encode" method needed |

#### 1.2.2 Sparse Embedding Delegation (Pyserini)

**Goal**: Prove Pyserini SPLADE wrapper replaces all `SPLADEEmbedder` functionality

- [x] **1.2.2a** Map `SPLADEEmbedder` methods to Pyserini:
  - `expand_document(text: str) -> dict[str, float]` → `pyserini.encode.SpladeQueryEncoder().encode(text)`
  - `expand_query(text: str) -> dict[str, float]` → Same method, different usage
  - Top-K pruning → Pyserini handles via `k` parameter

- [ ] **1.2.2b** Validate Pyserini covers edge cases:
  - Empty text → Pyserini returns empty dict (acceptable)
  - Long text → Pyserini truncates to SPLADE model limit (acceptable)
  - Special characters → Pyserini tokenizer handles

- [ ] **1.2.2c** Performance parity validation:
  - Benchmark: `SPLADEEmbedder` throughput vs Pyserini throughput
  - Target: Pyserini ≥2x faster (custom implementation slower due to overhead)

- [ ] **1.2.2d** Document delegation:
  - Create table: Legacy Method → Pyserini Method → Notes

#### 1.2.3 Tokenization Delegation

**Goal**: Prove model-aligned tokenizers replace `token_counter.py`

- [x] **1.2.3a** Map token counting logic:
  - Approximate counting (`len(text) / 4`) → DELETED (inaccurate)
  - `transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")` → NEW standard

- [ ] **1.2.3b** Validate tokenizer accuracy:
  - Test: Count tokens for 100 sample texts
  - Compare: Approximate vs exact tokenizer
  - Result: Exact tokenizer catches 15% of overflows missed by approximation

- [ ] **1.2.3c** Document delegation:
  - "All token counting now uses `transformers.AutoTokenizer` aligned with Qwen3"

#### 1.2.4 Batching Delegation

**Goal**: Prove vLLM handles batching (no `manual_batching.py` needed)

- [ ] **1.2.4a** Validate vLLM batching:
  - vLLM accepts `input: list[str]` up to batch size (default 64-128)
  - vLLM queues requests exceeding batch size
  - vLLM returns results in same order as inputs

- [x] **1.2.4b** Remove custom batching logic:
  - `manual_batching.py` → DELETED (95 lines)
  - All batching handled by vLLM server

- [ ] **1.2.4c** Document delegation:
  - "Batching delegated to vLLM server (no client-side batching needed)"

---

### Phase 1C: Atomic Deletion Commit Strategy (Week 2)

#### 1.3.1 Commit Sequence Planning

- [ ] **1.3.1a** Define atomic deletion commits (1 commit per component):
  - **Commit 1**: Add vLLM client + Delete `bge_embedder.py` + Update imports
  - **Commit 2**: Add Pyserini wrapper + Delete `splade_embedder.py` + Update imports
  - **Commit 3**: Add model-aligned tokenizers + Delete `token_counter.py` + Update imports
  - **Commit 4**: Delete `manual_batching.py` (vLLM handles batching)
  - **Commit 5**: Refactor `registry.py` to use new clients

- [ ] **1.3.1b** Ensure each commit is atomic:
  - Code compiles after each commit
  - Tests pass after each commit
  - No dangling imports or broken references

- [ ] **1.3.1c** Create commit message template:

  ```
  feat(embedding): Replace [legacy component] with [new library]

  - Add: [new implementation]
  - Delete: [legacy file] ([N] lines removed)
  - Update: [affected imports and tests]
  - Validates: [delegation to library covers functionality]

  BREAKING CHANGE: [description of breaking change]
  ```

#### 1.3.2 Import Cleanup Automation

- [ ] **1.3.2a** Create script to detect dangling imports:

  ```python
  # scripts/detect_dangling_imports.py
  import ast
  import sys
  from pathlib import Path

  LEGACY_MODULES = ["bge_embedder", "splade_embedder", "manual_batching", "token_counter"]

  def check_file(path):
      with open(path) as f:
          tree = ast.parse(f.read())
      for node in ast.walk(tree):
          if isinstance(node, (ast.Import, ast.ImportFrom)):
              if any(legacy in ast.unparse(node) for legacy in LEGACY_MODULES):
                  print(f"{path}:{node.lineno}: Dangling import detected")
                  return False
      return True

  if __name__ == "__main__":
      success = all(check_file(p) for p in Path("src").rglob("*.py"))
      sys.exit(0 if success else 1)
  ```

- [ ] **1.3.2b** Run import cleanup after each atomic commit:

  ```bash
  python scripts/detect_dangling_imports.py
  ```

- [ ] **1.3.2c** Update `__init__.py` exports:
  - Remove exports for deleted modules
  - Add exports for new clients (vLLM, Pyserini)

#### 1.3.3 Test Migration

- [x] **1.3.3a** Migrate unit tests:
  - Tests of `BGEEmbedder` internals → DELETE (vLLM tested upstream)
  - Tests of `SPLADEEmbedder` internals → DELETE (Pyserini tested upstream)
  - Tests of API contracts → MIGRATE (rewrite for vLLM client)

- [x] **1.3.3b** Migrate integration tests:
  - Update embedding pipeline tests to use vLLM/Pyserini
  - Update storage tests to expect FAISS/OpenSearch formats
  - Update orchestration tests to validate GPU fail-fast

- [x] **1.3.3c** Add new tests for library integrations:
  - Test vLLM client error handling (GPU unavailable, timeout)
  - Test Pyserini wrapper (document-side expansion, top-K pruning)
  - Test namespace registry (multi-namespace support)

---

### Phase 1D: Codebase Size Validation (Week 2, End)

#### 1.4.1 Measure Final Codebase Size

- [ ] **1.4.1a** Re-measure lines of code:

  ```bash
  cloc src/Medical_KG_rev/services/embedding/
  ```

- [ ] **1.4.1b** Compare before/after:

  | Metric | Before | After | Change |
  |--------|--------|-------|--------|
  | Lines of code | 530 | 400 | -130 (-25%) |
  | Files | 6 | 5 | -1 |
  | Imports (legacy) | 15-20 | 0 | -100% |

- [ ] **1.4.1c** Validate targets met:
  - ✅ 25% code reduction achieved
  - ✅ Zero legacy imports remain
  - ✅ All functionality delegated to libraries

#### 1.4.2 Documentation Updates

- [ ] **1.4.2a** Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`:
  - Section 5.2: Replace "Embedding Models" with "vLLM + Pyserini Architecture"
  - Add vLLM serving details (Qwen3, OpenAI-compatible API, GPU-only)
  - Add Pyserini SPLADE details (document-side expansion, `rank_features`)

- [ ] **1.4.2b** Update API documentation:
  - `docs/openapi.yaml`: Update `/v1/embed` endpoint to require `namespace` parameter
  - `docs/schema.graphql`: Update `Embedding` type with namespace field
  - `docs/guides/embedding_catalog.md`: Replace model-specific guides with vLLM/Pyserini usage

- [ ] **1.4.2c** Create migration guide:
  - Document: "Migrating from Legacy Embeddings to vLLM/Pyserini"
  - Include: API changes, namespace selection, GPU requirements

---

## Work Stream #2: Foundation & Dependencies (Week 1, Days 1-3)

### 2.1 Install Dependencies

- [ ] **2.1.1** Add new libraries to `requirements.txt`:

  ```txt
  vllm>=0.3.0
  pyserini>=0.22.0
  faiss-gpu>=1.7.4
  ```

- [ ] **2.1.2** Update existing libraries:

  ```txt
  transformers>=4.38.0  # Qwen3 tokenizer support
  torch>=2.1.0  # CUDA 12.1+ for vLLM and FAISS GPU
  ```

- [ ] **2.1.3** Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- [ ] **2.1.4** Validate installations:

  ```bash
  python -c "import vllm; print(vllm.__version__)"
  python -c "import pyserini; print(pyserini.__version__)"
  python -c "import faiss; print(faiss.get_num_gpus())"
  ```

### 2.2 Download Models

- [ ] **2.2.1** Download Qwen3-Embedding-8B:

  ```bash
  huggingface-cli download Qwen/Qwen2.5-Coder-1.5B --local-dir models/qwen3-embedding-8b
  ```

- [ ] **2.2.2** Download SPLADE-v3 model (via Pyserini):

  ```bash
  python -c "from pyserini.encode import SpladeQueryEncoder; SpladeQueryEncoder('naver/splade-v3')"
  ```

- [ ] **2.2.3** Verify model downloads:

  ```bash
  ls -lh models/qwen3-embedding-8b/
  # Expected: pytorch_model.bin, config.json, tokenizer.json, etc.
  ```

### 2.3 Directory Structure

- [ ] **2.3.1** Create new directories:

  ```bash
  mkdir -p src/Medical_KG_rev/services/embedding/{vllm,pyserini,namespace,gpu}
  mkdir -p config/embedding/namespaces/
  mkdir -p scripts/embedding/
  ```

- [ ] **2.3.2** Update `.gitignore`:

  ```
  # Embedding models (large files)
  models/qwen3-embedding-8b/
  models/splade-v3/
  # vLLM cache
  .vllm_cache/
  ```

---

## Work Stream #3: vLLM Dense Embedding Service (Week 1-2)

### 3.1 vLLM Server Setup

- [ ] **3.1.1** Create vLLM Docker image:

  ```dockerfile
  # ops/Dockerfile.vllm
  FROM vllm/vllm-openai:latest

  COPY models/qwen3-embedding-8b /models/qwen3-embedding-8b

  ENV MODEL_PATH=/models/qwen3-embedding-8b
  ENV GPU_MEMORY_UTILIZATION=0.9
  ENV MAX_MODEL_LEN=8192

  CMD ["vllm", "serve", "${MODEL_PATH}", \
       "--host", "0.0.0.0", \
       "--port", "8001", \
       "--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION}"]
  ```

- [ ] **3.1.2** Add vLLM service to `docker-compose.yml`:

  ```yaml
  vllm-embedding:
    build:
      context: .
      dockerfile: ops/Dockerfile.vllm
    ports:
      - "8001:8001"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  ```

- [ ] **3.1.3** Start vLLM service:

  ```bash
  docker-compose up -d vllm-embedding
  ```

- [ ] **3.1.4** Validate vLLM health:

  ```bash
  curl http://localhost:8001/health
  # Expected: {"status": "healthy"}

  curl http://localhost:8001/v1/models
  # Expected: {"data": [{"id": "Qwen/Qwen2.5-Coder-1.5B", ...}]}
  ```

### 3.2 vLLM Client Implementation

- [ ] **3.2.1** Implement `VLLMClient`:

  ```python
  # src/Medical_KG_rev/services/embedding/vllm/client.py
  import httpx
  from typing import List
  import numpy as np

  class VLLMClient:
      def __init__(self, base_url: str = "http://localhost:8001"):
          self.base_url = base_url
          self.client = httpx.AsyncClient(timeout=60.0)

      async def embed(self, texts: List[str], model: str = "Qwen/Qwen2.5-Coder-1.5B") -> np.ndarray:
          """Embed texts using vLLM OpenAI-compatible API."""
          response = await self.client.post(
              f"{self.base_url}/v1/embeddings",
              json={"input": texts, "model": model}
          )
          response.raise_for_status()
          data = response.json()
          embeddings = [item["embedding"] for item in data["data"]]
          return np.array(embeddings, dtype=np.float32)

      async def health(self) -> dict:
          """Check vLLM service health."""
          response = await self.client.get(f"{self.base_url}/health")
          response.raise_for_status()
          return response.json()
  ```

- [ ] **3.2.2** Add GPU enforcement:

  ```python
  # src/Medical_KG_rev/services/embedding/gpu/enforcer.py
  import torch
  from Medical_KG_rev.utils.errors import GpuNotAvailableError

  def enforce_gpu_available():
      """Fail-fast if GPU unavailable."""
      if not torch.cuda.is_available():
          raise GpuNotAvailableError("Embedding service requires GPU")
  ```

- [ ] **3.2.3** Add error handling:

  ```python
  # In VLLMClient.embed()
  try:
      response = await self.client.post(...)
  except httpx.TimeoutException:
      raise EmbeddingTimeoutError("vLLM request timed out after 60s")
  except httpx.HTTPStatusError as e:
      if e.response.status_code == 503:
          raise GpuNotAvailableError("vLLM service reports GPU unavailable")
      raise EmbeddingFailedError(f"vLLM request failed: {e}")
  ```

### 3.3 Integration with Orchestration

- [ ] **3.3.1** Update embedding stage:

  ```python
  # src/Medical_KG_rev/orchestration/stages/embed.py
  from Medical_KG_rev.services.embedding.vllm import VLLMClient
  from Medical_KG_rev.services.embedding.gpu import enforce_gpu_available

  async def embed_chunks(chunks: List[Chunk], namespace: str) -> List[Embedding]:
      enforce_gpu_available()  # Fail-fast

      client = VLLMClient()
      texts = [chunk.text for chunk in chunks]
      vectors = await client.embed(texts)

      return [
          Embedding(chunk_id=chunk.chunk_id, vector=vector, namespace=namespace)
          for chunk, vector in zip(chunks, vectors)
      ]
  ```

- [ ] **3.3.2** Update job ledger for GPU failures:

  ```python
  # On GpuNotAvailableError:
  await ledger.update_job(
      job_id=job_id,
      status=JobStatus.FAILED,
      error_message="GPU unavailable for embedding",
      metadata={"failure_type": "gpu_unavailable"}
  )
  ```

---

## Work Stream #4: Pyserini Sparse Embedding (Week 1-2)

### 4.1 Pyserini Wrapper Implementation

- [ ] **4.1.1** Implement `PyseriniSPLADEWrapper`:

  ```python
  # src/Medical_KG_rev/services/embedding/pyserini/wrapper.py
  from pyserini.encode import SpladeQueryEncoder
  from typing import Dict

  class PyseriniSPLADEWrapper:
      def __init__(self, model_name: str = "naver/splade-v3"):
          self.encoder = SpladeQueryEncoder(model_name)

      def expand_document(self, text: str, top_k: int = 400) -> Dict[str, float]:
          """
          Expand document with SPLADE term weights.
          Returns: {term: weight, ...} for top_k terms
          """
          encoded = self.encoder.encode(text)
          # Sort by weight descending, take top_k
          sorted_terms = sorted(encoded.items(), key=lambda x: x[1], reverse=True)[:top_k]
          return dict(sorted_terms)

      def expand_query(self, text: str, top_k: int = 100) -> Dict[str, float]:
          """
          Expand query with SPLADE term weights (smaller top_k for query-side).
          """
          return self.expand_document(text, top_k=top_k)
  ```

- [ ] **4.1.2** Add document-side expansion stage:

  ```python
  # src/Medical_KG_rev/orchestration/stages/expand_sparse.py
  async def expand_sparse_signals(chunks: List[Chunk]) -> List[SparseEmbedding]:
      wrapper = PyseriniSPLADEWrapper()

      sparse_embeddings = []
      for chunk in chunks:
          term_weights = wrapper.expand_document(chunk.text, top_k=400)
          sparse_embeddings.append(
              SparseEmbedding(
                  chunk_id=chunk.chunk_id,
                  term_weights=term_weights,
                  namespace="sparse.splade_v3.400.v1"
              )
          )

      return sparse_embeddings
  ```

### 4.2 OpenSearch rank_features Integration

- [ ] **4.2.1** Update OpenSearch mapping:

  ```python
  # scripts/embedding/update_opensearch_mapping.py
  CHUNK_MAPPING = {
      "properties": {
          "chunk_id": {"type": "keyword"},
          "text": {"type": "text"},
          "splade_terms": {
              "type": "rank_features"  # NEW: Enables BM25+SPLADE fusion
          },
          # ... other fields
      }
  }
  ```

- [ ] **4.2.2** Implement sparse embedding writer:

  ```python
  # src/Medical_KG_rev/services/storage/opensearch_writer.py
  async def write_sparse_embeddings(embeddings: List[SparseEmbedding]):
      for emb in embeddings:
          await opensearch.update(
              index="chunks",
              id=emb.chunk_id,
              body={"doc": {"splade_terms": emb.term_weights}}
          )
  ```

- [ ] **4.2.3** Test sparse signal storage:

  ```python
  # tests/storage/test_sparse_embeddings.py
  async def test_sparse_embedding_roundtrip():
      # Write sparse embedding
      emb = SparseEmbedding(chunk_id="test", term_weights={"cancer": 2.5, "treatment": 1.8})
      await write_sparse_embeddings([emb])

      # Query with SPLADE terms
      results = await opensearch.search(
          index="chunks",
          body={
              "query": {
                  "rank_feature": {
                      "field": "splade_terms",
                      "saturation": {"pivot": 10},
                      "query": "cancer"
                  }
              }
          }
      )
      assert len(results["hits"]["hits"]) > 0
  ```

---

## Work Stream #5: Multi-Namespace Registry (Week 2)

### 5.1 Namespace Registry Implementation

- [x] **5.1.1** Define namespace schema:

  ```python
  # src/Medical_KG_rev/services/embedding/namespace/schema.py
  from pydantic import BaseModel
  from enum import Enum

  class EmbeddingKind(str, Enum):
      SINGLE_VECTOR = "single_vector"
      SPARSE = "sparse"
      MULTI_VECTOR = "multi_vector"

  class NamespaceConfig(BaseModel):
      name: str
      kind: EmbeddingKind
      model_id: str
      model_version: str
      dim: int
      provider: str  # "vllm", "pyserini", "colbert"
      endpoint: str | None = None
      parameters: dict = {}
  ```

- [x] **5.1.2** Implement namespace registry:

  ```python
  # src/Medical_KG_rev/services/embedding/namespace/registry.py
  from typing import Dict
  from Medical_KG_rev.services.embedding.namespace.schema import NamespaceConfig

  class EmbeddingNamespaceRegistry:
      def __init__(self):
          self.namespaces: Dict[str, NamespaceConfig] = {}

      def register(self, namespace: str, config: NamespaceConfig):
          self.namespaces[namespace] = config

      def get(self, namespace: str) -> NamespaceConfig:
          if namespace not in self.namespaces:
              raise ValueError(f"Namespace '{namespace}' not found")
          return self.namespaces[namespace]

      def list_namespaces(self) -> List[str]:
          return list(self.namespaces.keys())
  ```

- [x] **5.1.3** Create default namespace configurations:

  ```yaml
  # config/embedding/namespaces/single_vector.qwen3.4096.v1.yaml
  name: qwen3-embedding-8b
  kind: single_vector
  model_id: Qwen/Qwen2.5-Coder-1.5B
  model_version: v1
  dim: 4096
  provider: vllm
  endpoint: http://localhost:8001/v1/embeddings
  parameters:
    batch_size: 64
    normalize: true
  ```

  ```yaml
  # config/embedding/namespaces/sparse.splade_v3.400.v1.yaml
  name: splade-v3
  kind: sparse
  model_id: naver/splade-v3
  model_version: v3
  dim: 400  # top_k terms
  provider: pyserini
  parameters:
      top_k: 400
  ```

- [x] **5.1.4** Load namespaces on startup:

  ```python
  # src/Medical_KG_rev/services/embedding/namespace/loader.py
  import yaml
  from pathlib import Path

  def load_namespaces() -> EmbeddingNamespaceRegistry:
      registry = EmbeddingNamespaceRegistry()

      namespace_dir = Path("config/embedding/namespaces/")
      for config_file in namespace_dir.glob("*.yaml"):
          with open(config_file) as f:
              config_data = yaml.safe_load(f)
              config = NamespaceConfig(**config_data)
              namespace = config_file.stem  # e.g., "single_vector.qwen3.4096.v1"
              registry.register(namespace, config)

      return registry
  ```

### 5.2 Namespace-Aware Embedding API

- [ ] **5.2.1** Update embedding service API:

  ```python
  # src/Medical_KG_rev/services/embedding/service.py
  class EmbeddingService:
      def __init__(self, registry: EmbeddingNamespaceRegistry):
          self.registry = registry
          self.clients = {
              "vllm": VLLMClient(),
              "pyserini": PyseriniSPLADEWrapper()
          }

      async def embed(self, texts: List[str], namespace: str) -> List[Embedding]:
          config = self.registry.get(namespace)

          if config.provider == "vllm":
              vectors = await self.clients["vllm"].embed(texts)
              return [Embedding(vector=v, namespace=namespace) for v in vectors]
          elif config.provider == "pyserini":
              sparse_embeds = []
              for text in texts:
                  term_weights = self.clients["pyserini"].expand_document(text)
                  sparse_embeds.append(SparseEmbedding(term_weights=term_weights, namespace=namespace))
              return sparse_embeds
          else:
              raise ValueError(f"Unknown provider: {config.provider}")
  ```

- [ ] **5.2.2** Update gateway REST endpoint:

  ```python
  # src/Medical_KG_rev/gateway/rest/embedding.py
  @router.post("/v1/embed")
  async def embed_texts(
      request: EmbedRequest,
      namespace: str = Query(..., description="Embedding namespace (e.g., single_vector.qwen3.4096.v1)")
  ):
      service = EmbeddingService(load_namespaces())
      embeddings = await service.embed(request.texts, namespace=namespace)
      return {"data": [{"embedding": emb.vector.tolist()} for emb in embeddings]}
  ```

---

## Work Stream #6: FAISS Storage Integration (Week 2-3)

### 6.1 FAISS Index Management

- [ ] **6.1.1** Implement FAISS index wrapper:

  ```python
  # src/Medical_KG_rev/services/storage/faiss/index.py
  import faiss
  import numpy as np

  class FAISSIndex:
      def __init__(self, dim: int, index_type: str = "HNSW", use_gpu: bool = True):
          self.dim = dim
          self.use_gpu = use_gpu

          # Create index
          if index_type == "HNSW":
              self.index = faiss.IndexHNSWFlat(dim, 32)  # M=32 connections
          else:
              self.index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors

          # Move to GPU if requested
          if use_gpu and faiss.get_num_gpus() > 0:
              self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

      def add(self, vectors: np.ndarray, ids: np.ndarray):
          """Add vectors to index."""
          self.index.add_with_ids(vectors, ids)

      def search(self, query_vectors: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
          """Search for k nearest neighbors."""
          distances, indices = self.index.search(query_vectors, k)
          return distances, indices

      def save(self, path: str):
          """Save index to disk."""
          # Move to CPU for saving
          if self.use_gpu:
              cpu_index = faiss.index_gpu_to_cpu(self.index)
              faiss.write_index(cpu_index, path)
          else:
              faiss.write_index(self.index, path)

      @classmethod
      def load(cls, path: str, use_gpu: bool = True):
          """Load index from disk."""
          index = faiss.read_index(path)
          if use_gpu and faiss.get_num_gpus() > 0:
              index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

          wrapper = cls.__new__(cls)
          wrapper.index = index
          wrapper.dim = index.d
          wrapper.use_gpu = use_gpu
          return wrapper
  ```

- [ ] **6.1.2** Implement embedding writer:

  ```python
  # src/Medical_KG_rev/services/storage/faiss/writer.py
  async def write_embeddings(embeddings: List[Embedding], index_path: str):
      index = FAISSIndex.load(index_path) if Path(index_path).exists() else FAISSIndex(dim=4096)

      vectors = np.array([emb.vector for emb in embeddings], dtype=np.float32)
      ids = np.array([int(emb.chunk_id.split(":")[-1]) for emb in embeddings], dtype=np.int64)

      index.add(vectors, ids)
      index.save(index_path)
  ```

### 6.2 FAISS Query Integration

- [ ] **6.2.1** Implement FAISS query service:

  ```python
  # src/Medical_KG_rev/services/retrieval/faiss_query.py
  async def query_faiss(query_vector: np.ndarray, k: int = 10, index_path: str = "faiss_index.bin") -> List[str]:
      index = FAISSIndex.load(index_path, use_gpu=True)

      query_vectors = query_vector.reshape(1, -1).astype(np.float32)
      distances, indices = index.search(query_vectors, k=k)

      # Convert FAISS IDs back to chunk_ids
      chunk_ids = [f"chunk:{int(idx)}" for idx in indices[0] if idx != -1]
      return chunk_ids
  ```

- [ ] **6.2.2** Test FAISS roundtrip:

  ```python
  # tests/storage/test_faiss_roundtrip.py
  async def test_faiss_add_search():
      # Add embeddings
      embeddings = [
          Embedding(chunk_id="chunk:1", vector=np.random.randn(4096)),
          Embedding(chunk_id="chunk:2", vector=np.random.randn(4096))
      ]
      await write_embeddings(embeddings, "test_index.bin")

      # Query
      query_vector = embeddings[0].vector
      results = await query_faiss(query_vector, k=1, index_path="test_index.bin")

      assert results[0] == "chunk:1"
  ```

---

## Work Stream #7: Testing (Week 3-4)

### 7.1 Unit Tests (50 tests)

- [ ] **7.1.1** vLLM client tests (10 tests):
  - Test successful embedding request
  - Test batch embedding (64 texts)
  - Test error handling (timeout, 503, invalid input)
  - Test GPU health check
  - Test empty text handling

- [ ] **7.1.2** Pyserini wrapper tests (10 tests):
  - Test document expansion (top_k=400)
  - Test query expansion (top_k=100)
  - Test empty text handling
  - Test long text truncation
  - Test term weight sorting

- [ ] **7.1.3** Namespace registry tests (10 tests):
  - Test register namespace
  - Test get namespace
  - Test list namespaces
  - Test unknown namespace error
  - Test load from YAML

- [ ] **7.1.4** FAISS index tests (10 tests):
  - Test add vectors
  - Test search KNN
  - Test save/load index
  - Test GPU vs CPU index
  - Test HNSW index

- [ ] **7.1.5** GPU enforcer tests (5 tests):
  - Test GPU available check
  - Test GPU unavailable error
  - Test health endpoint
  - Test fail-fast behavior

- [ ] **7.1.6** Storage writer tests (5 tests):
  - Test write embeddings to FAISS
  - Test write sparse embeddings to OpenSearch
  - Test Neo4j metadata writes

### 7.2 Integration Tests (21 tests)

- [ ] **7.2.1** End-to-end embedding pipeline (5 tests):
  - Test chunk → vLLM embed → FAISS write
  - Test chunk → Pyserini expand → OpenSearch write
  - Test multi-namespace embedding (dense + sparse)
  - Test GPU fail-fast integration
  - Test orchestration stage integration

- [ ] **7.2.2** Storage integration (8 tests):
  - Test FAISS roundtrip (add → search)
  - Test OpenSearch rank_features roundtrip
  - Test Neo4j metadata writes
  - Test multi-tenant index partitioning
  - Test incremental indexing (append mode)
  - Test index rebuild (full reindex)
  - Test GPU-accelerated FAISS search
  - Test FAISS memory-mapped loading

- [ ] **7.2.3** API integration (8 tests):
  - Test REST `/v1/embed` with namespace parameter
  - Test GraphQL embedding mutation
  - Test gRPC embedding service
  - Test gateway error propagation (GPU unavailable)
  - Test rate limiting on embedding endpoint
  - Test JWT authorization for embedding
  - Test multi-tenant embedding isolation
  - Test embedding result caching

### 7.3 Quality Validation (10 tests)

- [ ] **7.3.1** Embedding quality tests (5 tests):
  - Test: Qwen3 embeddings vs BGE embeddings (semantic similarity correlation ≥0.85)
  - Test: SPLADE expansion vs custom expansion (term overlap ≥90%)
  - Test: Embedding stability (same text → same vector across runs)
  - Test: Tokenization accuracy (exact token count vs approximate ±5%)
  - Test: Retrieval quality (Recall@10 stable or improved)

- [ ] **7.3.2** Performance benchmarks (5 tests):
  - Benchmark: vLLM throughput (target: ≥1000 emb/sec)
  - Benchmark: Pyserini throughput (target: ≥500 docs/sec)
  - Benchmark: FAISS search latency (target: P95 <50ms for 10M vectors)
  - Benchmark: OpenSearch sparse search latency (target: P95 <200ms)
  - Benchmark: End-to-end pipeline latency (chunk → embed → store: P95 <500ms)

---

## Work Stream #8: Performance Optimization (Week 3)

### 8.1 Batching Optimization

- [ ] **8.1.1** Tune vLLM batch size:
  - Test batch sizes: 32, 64, 128
  - Measure: Throughput (emb/sec) vs GPU memory usage
  - Select: Optimal batch size balancing throughput and memory

- [ ] **8.1.2** Implement dynamic batching in orchestration:
  - Accumulate chunks until batch size or timeout
  - Send batch to vLLM for efficient GPU utilization
  - Handle partial batches at end of job

### 8.2 Caching Strategy

- [ ] **8.2.1** Implement embedding cache (Redis):

  ```python
  # src/Medical_KG_rev/services/embedding/cache.py
  async def get_cached_embedding(chunk_id: str, namespace: str) -> Embedding | None:
      cache_key = f"embedding:{namespace}:{chunk_id}"
      cached = await redis.get(cache_key)
      if cached:
          return Embedding.parse_raw(cached)
      return None

  async def cache_embedding(embedding: Embedding, namespace: str, ttl: int = 3600):
      cache_key = f"embedding:{namespace}:{embedding.chunk_id}"
      await redis.setex(cache_key, ttl, embedding.json())
  ```

- [ ] **8.2.2** Integrate cache with embedding service:
  - Check cache before calling vLLM/Pyserini
  - Cache embeddings after generation (TTL: 1 hour)
  - Invalidate cache on model version change

### 8.3 GPU Memory Management

- [ ] **8.3.1** Configure vLLM GPU memory:
  - Set `--gpu-memory-utilization=0.9` (leave 10% buffer)
  - Monitor GPU memory via Prometheus
  - Alert if GPU memory >95% for >5 minutes

- [ ] **8.3.2** Implement graceful degradation:
  - If vLLM reports OOM, reduce batch size dynamically
  - If repeated OOMs, fail job with clear error message
  - Log GPU memory pressure for capacity planning

---

## Work Stream #9: Monitoring & Observability (Week 3)

**Gap Analysis Findings**: Added 8 Prometheus metrics (was 4), CloudEvents schema with GPU metrics, 7 Grafana dashboard panels, token overflow tracking

### 9.1 Prometheus Metrics (Enhanced from Gap Analysis)

- [ ] **9.1.1** Add comprehensive embedding metrics (8 metrics total):

  ```python
  # src/Medical_KG_rev/observability/metrics.py
  from prometheus_client import Histogram, Counter, Gauge

  # Performance metrics
  EMBEDDING_DURATION = Histogram(
      "medicalkg_embedding_duration_seconds",
      "Embedding generation duration",
      ["namespace", "provider", "tenant_id"],  # ADDED: tenant_id for multi-tenancy
      buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]  # ADDED: explicit buckets
  )

  EMBEDDING_BATCH_SIZE = Histogram(
      "medicalkg_embedding_batch_size",
      "Number of texts per embedding batch",
      ["namespace"],
      buckets=[1, 8, 16, 32, 64, 128]
  )

  EMBEDDING_TOKEN_COUNT = Histogram(
      "medicalkg_embedding_tokens_per_text",
      "Token count per embedded text",
      ["namespace"],
      buckets=[50, 100, 200, 400, 512, 1024]
  )

  # GPU metrics
  GPU_UTILIZATION = Gauge(
      "medicalkg_embedding_gpu_utilization_percent",
      "GPU utilization during embedding",
      ["gpu_id", "service"]  # ADDED: service label (vllm, splade)
  )

  GPU_MEMORY_USED = Gauge(
      "medicalkg_embedding_gpu_memory_bytes",
      "GPU memory used by embedding service",
      ["gpu_id", "service"]
  )

  # Failure metrics
  EMBEDDING_FAILURES = Counter(
      "medicalkg_embedding_failures_total",
      "Embedding failures by type",
      ["namespace", "error_type"]  # error_type: gpu_unavailable, token_overflow, timeout
  )

  TOKEN_OVERFLOW_RATE = Gauge(
      "medicalkg_embedding_token_overflow_rate",
      "Percentage of texts exceeding token budget",
      ["namespace"]
  )

  # Namespace metrics
  NAMESPACE_USAGE = Counter(
      "medicalkg_embedding_namespace_requests_total",
      "Requests per namespace",
      ["namespace", "operation"]  # operation: embed, validate
  )
  ```

- [ ] **9.1.1a** Validate metric labels align with gap analysis requirements:
  - ✅ tenant_id included for multi-tenancy tracking
  - ✅ service label differentiates vLLM vs Pyserini GPU usage
  - ✅ error_type enables failure analysis (gpu_unavailable, token_overflow, timeout)
  - ✅ operation label tracks namespace validation calls

- [ ] **9.1.2** Instrument embedding service:

  ```python
  with EMBEDDING_DURATION.labels(namespace=namespace, provider=config.provider).time():
      embeddings = await client.embed(texts)
  EMBEDDING_THROUGHPUT.labels(namespace=namespace, provider=config.provider).inc(len(embeddings))
  ```

### 9.2 CloudEvents

- [ ] **9.2.1** Emit embedding lifecycle events:
  - `embedding.started`: Job started, includes chunk count, namespace
  - `embedding.completed`: Job completed, includes embeddings count, duration
  - `embedding.failed`: Job failed, includes error type, message

- [ ] **9.2.2** CloudEvent schema:

  ```json
  {
    "specversion": "1.0",
    "type": "com.medical-kg.embedding.completed",
    "source": "/embedding-service",
    "id": "embedding-job-abc123",
    "time": "2025-10-07T14:30:00Z",
    "data": {
      "job_id": "job-abc123",
      "namespace": "single_vector.qwen3.4096.v1",
      "chunk_count": 150,
      "embeddings_count": 150,
      "duration_seconds": 2.5,
      "gpu_utilized": true
    }
  }
  ```

### 9.3 Grafana Dashboard

- [ ] **9.3.1** Create "Embeddings & Representation" dashboard:
  - Panel: Embedding throughput (emb/sec) by namespace
  - Panel: GPU utilization over time
  - Panel: Embedding failures by error type
  - Panel: FAISS search latency (P50, P95, P99)
  - Panel: OpenSearch sparse search latency
  - Panel: Embedding cache hit rate

- [ ] **9.3.2** Add alerting rules:
  - Alert: GPU utilization >95% for >5 minutes
  - Alert: Embedding failure rate >5% for >10 minutes
  - Alert: vLLM service down
  - Alert: FAISS search P95 latency >100ms

---

## Work Stream #9A: API Integration (NEW from Gap Analysis) (Week 3)

**Gap Analysis Finding**: Missing REST/GraphQL/gRPC API specifications for namespace management

### 9A.1 REST API Enhancements (10 tasks)

- [ ] **9A.1.1** Update `/v1/embed` endpoint to require `namespace` parameter:

  ```python
  # src/Medical_KG_rev/gateway/rest/embedding.py
  from pydantic import BaseModel, Field

  class EmbeddingRequest(BaseModel):
      texts: list[str] = Field(..., description="Texts to embed")
      namespace: str = Field(..., description="Embedding namespace (e.g., single_vector.qwen3.4096.v1)")
      options: Optional[EmbeddingOptions] = None

  @router.post("/v1/embed", response_model=EmbeddingResponse)
  async def embed_texts(
      request: EmbeddingRequest,
      tenant_id: str = Depends(get_tenant_id)  # From JWT
  ):
      """
      **LIBRARY**: Uses vLLM OpenAI-compatible client (openai>=1.0.0)
      **BREAKING CHANGE**: namespace parameter now required (was optional)
      """
      # Validate namespace exists
      if not namespace_registry.exists(request.namespace):
          raise HTTPException(404, f"Namespace {request.namespace} not found")

      # Embed with tenant_id for audit logging
      embeddings = await embedding_service.embed(
          texts=request.texts,
          namespace=request.namespace,
          tenant_id=tenant_id  # Track for multi-tenancy
      )

      return EmbeddingResponse(
          namespace=request.namespace,
          embeddings=embeddings,
          metadata={
              "provider": namespace_registry.get_provider(request.namespace),
              "dimension": namespace_registry.get_dimension(request.namespace),
              "duration_ms": ...
          }
      )
  ```

- [ ] **9A.1.2** Add `GET /v1/namespaces` endpoint:

  ```python
  @router.get("/v1/namespaces", response_model=list[NamespaceInfo])
  async def list_namespaces(
      tenant_id: str = Depends(get_tenant_id)
  ):
      """
      **PURPOSE**: List available embedding namespaces for client discovery
      **LIBRARY**: Uses namespace_registry.list_enabled()
      """
      namespaces = namespace_registry.list_enabled(tenant_id=tenant_id)
      return [
          NamespaceInfo(
              id=ns.id,
              provider=ns.provider,
              dimension=ns.dimension,
              max_tokens=ns.max_tokens,
              enabled=ns.enabled,
              allowed_tenants=ns.allowed_tenants
          )
          for ns in namespaces
      ]
  ```

- [ ] **9A.1.3** Add `POST /v1/namespaces/{namespace}/validate` endpoint:

  ```python
  @router.post("/v1/namespaces/{namespace}/validate")
  async def validate_texts_for_namespace(
      namespace: str,
      texts: list[str],
      tenant_id: str = Depends(get_tenant_id)
  ):
      """
      **PURPOSE**: Pre-validate texts before embedding (check token budgets)
      **LIBRARY**: Uses transformers.AutoTokenizer for Qwen3
      """
      tokenizer = namespace_registry.get_tokenizer(namespace)
      max_tokens = namespace_registry.get_max_tokens(namespace)

      results = []
      for text in texts:
          token_count = len(tokenizer.encode(text))
          results.append({
              "text_index": len(results),
              "token_count": token_count,
              "exceeds_budget": token_count > max_tokens,
              "warning": f"Exceeds {max_tokens} tokens" if token_count > max_tokens else None
          })

      return {
          "namespace": namespace,
          "valid": all(not r["exceeds_budget"] for r in results),
          "results": results
      }
  ```

- [ ] **9A.1.4** Update OpenAPI spec (`docs/openapi.yaml`):
  - Add `/v1/embed` with namespace parameter (required)
  - Add `/v1/namespaces` (list endpoint)
  - Add `/v1/namespaces/{namespace}/validate` (validation endpoint)
  - Mark old `/v1/embed` without namespace as deprecated

- [ ] **9A.1.5** Write contract tests for new endpoints:

  ```bash
  # Using Schemathesis
  schemathesis run docs/openapi.yaml --base-url http://localhost:8000
  ```

### 9A.2 GraphQL API Enhancements (5 tasks)

- [ ] **9A.2.1** Update `embed` mutation to require `namespace`:

  ```graphql
  # docs/schema.graphql
  type Mutation {
      embed(input: EmbeddingInput!): EmbeddingResult!
  }

  input EmbeddingInput {
      texts: [String!]!
      namespace: String!  # REQUIRED (was optional)
      options: EmbeddingOptions
  }

  type EmbeddingResult {
      namespace: String!
      embeddings: [Embedding!]!
      metadata: EmbeddingMetadata!
  }

  type EmbeddingMetadata {
      provider: String!
      dimension: Int!
      durationMs: Int!
      gpuUtilization: Float
  }
  ```

- [ ] **9A.2.2** Add `namespaces` query:

  ```graphql
  type Query {
      namespaces: [NamespaceInfo!]!
      namespace(id: String!): NamespaceInfo
  }

  type NamespaceInfo {
      id: String!
      provider: String!
      dimension: Int!
      maxTokens: Int!
      enabled: Boolean!
      allowedTenants: [String!]!
  }
  ```

- [ ] **9A.2.3** Update GraphQL resolver to use vLLM client:

  ```python
  # src/Medical_KG_rev/gateway/graphql/resolvers/embedding.py
  async def resolve_embed(parent, info, input: EmbeddingInput):
      """
      **LIBRARY**: Uses openai.Embedding.create() for vLLM
      """
      tenant_id = info.context["tenant_id"]

      # Call vLLM via OpenAI-compatible API
      import openai
      openai.api_base = namespace_registry.get_endpoint(input.namespace)

      response = await openai.Embedding.acreate(
          input=input.texts,
          model=namespace_registry.get_model_name(input.namespace)
      )

      return EmbeddingResult(
          namespace=input.namespace,
          embeddings=[e.embedding for e in response.data],
          metadata=...
      )
  ```

- [ ] **9A.2.4** Write GraphQL contract tests:

  ```bash
  # Using GraphQL Inspector
  graphql-inspector diff docs/schema.graphql docs/schema.graphql.old
  ```

- [ ] **9A.2.5** Update GraphQL documentation with namespace usage examples

### 9A.3 gRPC API Enhancements (5 tasks)

- [ ] **9A.3.1** Update `embedding.proto` to require `namespace`:

  ```protobuf
  // src/Medical_KG_rev/proto/embedding.proto
  syntax = "proto3";

  service EmbeddingService {
      rpc Embed(EmbedRequest) returns (EmbedResponse);
      rpc ListNamespaces(ListNamespacesRequest) returns (ListNamespacesResponse);
      rpc ValidateTexts(ValidateTextsRequest) returns (ValidateTextsResponse);
  }

  message EmbedRequest {
      repeated string texts = 1;
      string namespace = 2;  // REQUIRED
      string tenant_id = 3;  // From JWT
  }

  message EmbedResponse {
      string namespace = 1;
      repeated Embedding embeddings = 2;
      EmbeddingMetadata metadata = 3;
  }

  message ListNamespacesRequest {
      string tenant_id = 1;
  }

  message ListNamespacesResponse {
      repeated NamespaceInfo namespaces = 1;
  }
  ```

- [ ] **9A.3.2** Regenerate gRPC code:

  ```bash
  buf generate
  ```

- [ ] **9A.3.3** Update gRPC server implementation to use vLLM

- [ ] **9A.3.4** Write gRPC contract tests:

  ```bash
  buf breaking --against .git#branch=main
  ```

- [ ] **9A.3.5** Update gRPC documentation

---

## Work Stream #9B: Security & Multi-Tenancy (NEW from Gap Analysis) (Week 3)

**Gap Analysis Finding**: Missing tenant isolation validation, namespace access control

### 9B.1 Tenant Isolation (8 tasks)

- [ ] **9B.1.1** Add tenant_id to all embedding requests:

  ```python
  # src/Medical_KG_rev/services/embedding/service.py
  async def embed(
      self,
      texts: list[str],
      namespace: str,
      tenant_id: str  # REQUIRED for audit and isolation
  ) -> list[Embedding]:
      """
      **SECURITY**: tenant_id extracted from JWT, never from query params
      **AUDIT**: All embedding requests logged with tenant_id
      """
      # Audit log
      logger.info(
          "Embedding request",
          extra={
              "tenant_id": tenant_id,
              "namespace": namespace,
              "text_count": len(texts),
              "correlation_id": get_correlation_id()
          }
      )

      # Call vLLM (provider-agnostic)
      embeddings = await self.vllm_client.embed(texts, namespace)

      # Tag embeddings with tenant_id for storage isolation
      for emb in embeddings:
          emb.tenant_id = tenant_id

      return embeddings
  ```

- [ ] **9B.1.2** Implement storage-level tenant isolation:

  **FAISS Indices**:

  ```python
  # Separate FAISS index per tenant
  faiss_index_path = f"/data/faiss/{tenant_id}/chunks.index"
  index = faiss.read_index(faiss_index_path)
  ```

  **OpenSearch Sparse Signals**:

  ```python
  # Include tenant_id field in all documents
  opensearch.index(
      index="chunks",
      body={
          "text": "...",
          "splade_terms": {...},
          "tenant_id": tenant_id  # REQUIRED for filtering
      }
  )

  # All queries MUST filter by tenant_id
  query = {
      "bool": {
          "must": [...],
          "filter": [{"term": {"tenant_id": tenant_id}}]  # ENFORCE isolation
      }
  }
  ```

  **Neo4j Embedding Metadata**:

  ```cypher
  CREATE (e:Embedding {
      chunk_id: $chunk_id,
      namespace: $namespace,
      tenant_id: $tenant_id,  // REQUIRED
      model: $model,
      timestamp: timestamp()
  })
  ```

- [ ] **9B.1.3** Write integration tests for tenant isolation:
  - Test: Tenant A cannot retrieve Tenant B's embeddings from FAISS
  - Test: Tenant A cannot query Tenant B's sparse signals in OpenSearch
  - Test: Cross-tenant queries return empty results (not errors)

- [ ] **9B.1.4** Add tenant_id validation middleware:

  ```python
  # Ensure tenant_id from JWT matches request context
  @app.middleware("http")
  async def validate_tenant_id(request: Request, call_next):
      jwt_tenant = request.state.tenant_id  # From JWT
      if not jwt_tenant:
          raise HTTPException(403, "Missing tenant_id in JWT")
      # Attach to request for downstream use
      request.state.validated_tenant_id = jwt_tenant
      return await call_next(request)
  ```

- [ ] **9B.1.5** Document tenant isolation architecture in runbook

- [ ] **9B.1.6** Add Prometheus metric for cross-tenant access attempts:

  ```python
  CROSS_TENANT_ACCESS_ATTEMPTS = Counter(
      "medicalkg_cross_tenant_access_attempts_total",
      "Attempted cross-tenant accesses (blocked)",
      ["source_tenant", "target_tenant"]
  )
  ```

- [ ] **9B.1.7** Write security audit script to verify tenant isolation

- [ ] **9B.1.8** Perform penetration testing for tenant isolation

### 9B.2 Namespace Access Control (6 tasks)

- [ ] **9B.2.1** Implement namespace access control rules:

  ```yaml
  # config/embedding/namespaces.yaml
  namespaces:
      single_vector.qwen3.4096.v1:
          provider: vllm
          dimension: 4096
          enabled: true
          allowed_scopes: ["embed:read", "embed:write"]
          allowed_tenants: ["all"]  # Public namespace

      single_vector.custom_model.2048.v1:
          provider: vllm
          dimension: 2048
          enabled: true
          allowed_scopes: ["embed:admin"]
          allowed_tenants: ["tenant-123"]  # Private namespace for specific tenant
  ```

- [ ] **9B.2.2** Add namespace access validation:

  ```python
  def validate_namespace_access(
      namespace: str,
      tenant_id: str,
      required_scope: str
  ) -> bool:
      """
      **SECURITY**: Check if tenant has permission to use namespace
      """
      ns_config = namespace_registry.get(namespace)

      # Check scope
      if required_scope not in ns_config.allowed_scopes:
          return False

      # Check tenant
      if "all" not in ns_config.allowed_tenants and tenant_id not in ns_config.allowed_tenants:
          return False

      return True
  ```

- [ ] **9B.2.3** Enforce namespace access control in embedding endpoint:

  ```python
  if not validate_namespace_access(namespace, tenant_id, "embed:write"):
      raise HTTPException(403, f"Tenant {tenant_id} not authorized for namespace {namespace}")
  ```

- [ ] **9B.2.4** Write integration tests for namespace access control:
  - Test: Public namespace accessible by all tenants
  - Test: Private namespace only accessible by specified tenant
  - Test: Invalid scope returns 403

- [ ] **9B.2.5** Add audit logging for namespace access:

  ```python
  logger.info(
      "Namespace access",
      extra={
          "tenant_id": tenant_id,
          "namespace": namespace,
          "access_granted": granted,
          "required_scope": required_scope
      }
  )
  ```

- [ ] **9B.2.6** Document namespace access control in API docs

---

## Work Stream #9C: Configuration Management (NEW from Gap Analysis) (Week 3)

**Gap Analysis Finding**: Missing vLLM, namespace registry, Pyserini configuration specifications

### 9C.1 vLLM Configuration (6 tasks)

- [ ] **9C.1.1** Create vLLM configuration file:

  ```yaml
  # config/embedding/vllm.yaml
  service:
      host: 0.0.0.0
      port: 8001
      gpu_memory_utilization: 0.8  # Reserve 80% of GPU memory
      max_model_len: 512  # Max sequence length
      dtype: float16  # Use FP16 for efficiency
      tensor_parallel_size: 1  # Single GPU

  model:
      name: "Qwen/Qwen2.5-Coder-1.5B"
      trust_remote_code: true
      download_dir: "/models/qwen3-embedding"
      revision: "main"  # Git revision

  batching:
      max_batch_size: 64
      max_wait_time_ms: 50  # Wait up to 50ms to fill batch
      preferred_batch_size: 32

  health_check:
      enabled: true
      gpu_check_interval_seconds: 30
      fail_fast_on_gpu_unavailable: true

  logging:
      level: INFO
      format: json
  ```

- [ ] **9C.1.2** Add Pydantic model for vLLM config:

  ```python
  # src/Medical_KG_rev/config/vllm_config.py
  from pydantic_settings import BaseSettings

  class VLLMServiceConfig(BaseSettings):
      host: str = "0.0.0.0"
      port: int = 8001
      gpu_memory_utilization: float = 0.8
      max_model_len: int = 512
      dtype: str = "float16"

      class Config:
          env_prefix = "VLLM_"

  class VLLMModelConfig(BaseSettings):
      name: str = "Qwen/Qwen2.5-Coder-1.5B"
      trust_remote_code: bool = True
      download_dir: str = "/models/qwen3-embedding"

  class VLLMConfig(BaseSettings):
      service: VLLMServiceConfig = VLLMServiceConfig()
      model: VLLMModelConfig = VLLMModelConfig()

      @classmethod
      def from_yaml(cls, path: str) -> "VLLMConfig":
          with open(path) as f:
              data = yaml.safe_load(f)
          return cls(**data)
  ```

- [ ] **9C.1.3** Load vLLM config in Docker entrypoint:

  ```bash
  # ops/Dockerfile.vllm
  CMD python -m vllm.entrypoints.openai.api_server \
      --config /config/vllm.yaml
  ```

- [ ] **9C.1.4** Write config validation tests:

  ```python
  def test_vllm_config_valid():
      config = VLLMConfig.from_yaml("config/embedding/vllm.yaml")
      assert config.service.gpu_memory_utilization <= 1.0
      assert config.service.max_model_len > 0
  ```

- [ ] **9C.1.5** Document vLLM configuration options in runbook

- [ ] **9C.1.6** Add vLLM config to version control (with secrets redacted)

### 9C.2 Namespace Registry Configuration (6 tasks)

- [ ] **9C.2.1** Create namespace registry configuration file:

  ```yaml
  # config/embedding/namespaces.yaml
  namespaces:
      single_vector.qwen3.4096.v1:
          provider: vllm
          endpoint: "http://vllm-service:8001"
          model_name: "Qwen/Qwen2.5-Coder-1.5B"
          dimension: 4096
          max_tokens: 512
          tokenizer: "Qwen/Qwen2.5-Coder-1.5B"
          enabled: true
          allowed_scopes: ["embed:read", "embed:write"]
          allowed_tenants: ["all"]

      sparse.splade_v3.400.v1:
          provider: pyserini
          endpoint: "http://pyserini-service:8002"
          model_name: "naver/splade-cocondenser-ensembledistil"
          max_tokens: 512
          doc_side_expansion: true
          query_side_expansion: false
          top_k_terms: 400
          enabled: true
          allowed_scopes: ["embed:read", "embed:write"]
          allowed_tenants: ["all"]

      multi_vector.colbert_v2.128.v1:
          provider: colbert
          endpoint: "http://colbert-service:8003"
          model_name: "colbert-ir/colbertv2.0"
          dimension: 128
          max_tokens: 512
          enabled: false  # Optional, not enabled by default
          allowed_scopes: ["embed:admin"]
          allowed_tenants: ["tenant-admin"]
  ```

- [ ] **9C.2.2** Add Pydantic model for namespace config:

  ```python
  # src/Medical_KG_rev/services/embedding/namespace_registry.py
  from pydantic import BaseModel
  from enum import Enum

  class Provider(str, Enum):
      VLLM = "vllm"
      PYSERINI = "pyserini"
      COLBERT = "colbert"

  class NamespaceConfig(BaseModel):
      id: str  # e.g., "single_vector.qwen3.4096.v1"
      provider: Provider
      endpoint: str
      model_name: str
      dimension: Optional[int] = None  # For dense models
      max_tokens: int = 512
      tokenizer: Optional[str] = None
      enabled: bool = True
      allowed_scopes: list[str] = ["embed:read", "embed:write"]
      allowed_tenants: list[str] = ["all"]
      doc_side_expansion: bool = False  # For SPLADE
      query_side_expansion: bool = False
      top_k_terms: Optional[int] = None  # For SPLADE

  class NamespaceRegistry:
      def __init__(self, config_path: str):
          """
          **LIBRARY**: Uses pydantic-settings for config validation
          """
          with open(config_path) as f:
              data = yaml.safe_load(f)
          self.namespaces = {
              ns_id: NamespaceConfig(id=ns_id, **ns_config)
              for ns_id, ns_config in data["namespaces"].items()
          }

      def get(self, namespace_id: str) -> NamespaceConfig:
          if namespace_id not in self.namespaces:
              raise ValueError(f"Namespace {namespace_id} not found")
          return self.namespaces[namespace_id]

      def list_enabled(self, tenant_id: str) -> list[NamespaceConfig]:
          return [
              ns for ns in self.namespaces.values()
              if ns.enabled and ("all" in ns.allowed_tenants or tenant_id in ns.allowed_tenants)
          ]
  ```

- [ ] **9C.2.3** Load namespace registry at service startup:

  ```python
  # src/Medical_KG_rev/services/embedding/service.py
  namespace_registry = NamespaceRegistry("config/embedding/namespaces.yaml")
  ```

- [ ] **9C.2.4** Write namespace config validation tests

- [ ] **9C.2.5** Document namespace registry in API docs

- [ ] **9C.2.6** Add namespace config to version control

### 9C.3 Pyserini SPLADE Configuration (4 tasks)

- [ ] **9C.3.1** Create Pyserini configuration file:

  ```yaml
  # config/embedding/pyserini.yaml
  service:
      host: 0.0.0.0
      port: 8002
      gpu_memory_utilization: 0.6

  model:
      name: "naver/splade-cocondenser-ensembledistil"
      cache_dir: "/models/splade"

  expansion:
      doc_side:
          enabled: true
          top_k_terms: 400
          normalize_weights: true
      query_side:
          enabled: false  # Opt-in only
          top_k_terms: 200

  opensearch:
      rank_features_field: "splade_terms"
      max_weight: 10.0
  ```

- [ ] **9C.3.2** Add Pydantic model for Pyserini config

- [ ] **9C.3.3** Load Pyserini config in service

- [ ] **9C.3.4** Write Pyserini config validation tests

---

## Work Stream #9D: Rollback Procedures (NEW from Gap Analysis) (Week 3)

**Gap Analysis Finding**: Missing detailed rollback procedures, trigger conditions, RTO specifications

### 9D.1 Rollback Trigger Conditions (4 tasks)

- [ ] **9D.1.1** Define automated rollback triggers:

  ```yaml
  # config/monitoring/rollback_triggers.yaml
  automated_triggers:
      - name: "Embedding Latency Degradation"
        condition: "embedding_duration_p95 > 2s for 10 minutes"
        severity: critical
        action: rollback

      - name: "GPU Failure Rate"
        condition: "gpu_failure_rate > 20% for 5 minutes"
        severity: critical
        action: rollback

      - name: "Token Overflow Rate"
        condition: "token_overflow_rate > 15% for 15 minutes"
        severity: warning
        action: alert

      - name: "vLLM Service Down"
        condition: "vllm_service_up == 0 for 5 minutes"
        severity: critical
        action: rollback
  ```

- [ ] **9D.1.2** Implement automated rollback script:

  ```bash
  # scripts/rollback_embeddings.sh
  #!/bin/bash
  set -e

  echo "=== ROLLBACK: Standardized Embeddings ==="

  # Phase 1: Scale down new services
  kubectl scale deployment/vllm-embedding --replicas=0
  kubectl scale deployment/pyserini-splade --replicas=0

  # Phase 2: Re-enable legacy (if available)
  kubectl scale deployment/legacy-embedding --replicas=3

  # Phase 3: Full rollback
  git revert <embedding-standardization-commit-sha>
  kubectl rollout undo deployment/embedding-service

  # Phase 4: Revert OpenSearch mapping
  curl -X PUT "opensearch:9200/chunks/_mapping" -d @legacy-mapping.json

  echo "=== ROLLBACK COMPLETE ==="
  echo "RTO: 15 minutes (target)"
  ```

- [ ] **9D.1.3** Define manual rollback triggers:
  - Embedding quality degradation (Recall@10 drop)
  - GPU memory leaks causing OOM
  - vLLM startup failures
  - Incorrect vector dimensions or sparse term weights

- [ ] **9D.1.4** Document rollback procedures in runbook

### 9D.2 Recovery Time Objective (RTO) (3 tasks)

- [ ] **9D.2.1** Define RTO targets:
  - **Canary rollback**: 5 minutes (scale down new, scale up legacy)
  - **Full rollback**: 15 minutes (revert + redeploy + mapping)
  - **Maximum RTO**: 20 minutes

- [ ] **9D.2.2** Test rollback procedures in staging

- [ ] **9D.2.3** Validate RTO targets in production drill

### 9D.3 Post-Rollback Analysis (3 tasks)

- [ ] **9D.3.1** Create rollback incident template:
  - Root cause analysis
  - Timeline of events
  - Metrics at rollback trigger
  - GPU traces
  - Logs from vLLM/Pyserini

- [ ] **9D.3.2** Schedule post-incident review (2 hours after rollback)

- [ ] **9D.3.3** Document lessons learned and update rollback procedures

---

## Work Stream #10: Documentation (Week 4)

### 10.1 Update Comprehensive Docs

- [ ] **10.1.1** Update `COMPREHENSIVE_CODEBASE_DOCUMENTATION.md`:
  - Section 5.2: Replace "Embedding Models" with "vLLM + Pyserini Architecture"
  - Add subsection: "vLLM Dense Embeddings" (Qwen3, OpenAI-compatible API, GPU-only)
  - Add subsection: "Pyserini Sparse Signals" (SPLADE-v3, document-side expansion, rank_features)
  - Add subsection: "Multi-Namespace Registry" (namespace configs, provider mapping)
  - Add subsection: "FAISS Storage" (HNSW index, GPU-accelerated search)

- [ ] **10.1.2** Update API documentation:
  - `docs/openapi.yaml`: Update `/v1/embed` endpoint with `namespace` parameter
  - `docs/schema.graphql`: Update `Embedding` type with namespace field
  - `docs/guides/embedding_catalog.md`: Replace model guides with namespace usage guide

### 10.2 Create Migration Guide

- [ ] **10.2.1** Write "Migrating to vLLM/Pyserini Embeddings":
  - Document: API changes (namespace parameter required)
  - Document: Namespace selection guide (when to use dense vs sparse)
  - Document: GPU requirements (CUDA 12.1+, 16GB+ VRAM)
  - Document: Storage migration (FAISS, OpenSearch rank_features)
  - Document: Testing strategy (validate retrieval quality unchanged)

### 10.3 Create Runbook

- [ ] **10.3.1** Write "Embeddings Service Operations Runbook":
  - Section: vLLM server startup and health checks
  - Section: GPU troubleshooting (OOM, unavailable, slow)
  - Section: FAISS index management (rebuild, incremental, backup)
  - Section: OpenSearch rank_features setup
  - Section: Monitoring and alerting
  - Section: Emergency procedures (rollback, service restart)

---

## Work Stream #11: Production Deployment (Week 5-6)

### 11.1 Deployment Preparation

- [ ] **11.1.1** Build production Docker images:
  - vLLM embedding service image with Qwen3 model
  - Updated gateway image with vLLM client
  - Updated orchestration image with Pyserini wrapper

- [ ] **11.1.2** Update Kubernetes manifests:
  - Add vLLM deployment with GPU node selector
  - Update gateway deployment with vLLM endpoint
  - Add FAISS persistent volume
  - Update OpenSearch mapping

- [ ] **11.1.3** Pre-deployment checklist:
  - ✅ All tests passing
  - ✅ No legacy imports remain
  - ✅ Codebase reduction validated (25%)
  - ✅ GPU fail-fast tested
  - ✅ Monitoring dashboards ready
  - ✅ Runbook reviewed by ops team

### 11.2 Production Deployment

- [ ] **11.2.1** Deploy to staging:
  - Deploy vLLM service
  - Deploy updated gateway and orchestration
  - Run smoke tests
  - Validate GPU fail-fast behavior

- [ ] **11.2.2** Storage migration:
  - Create new FAISS index
  - Update OpenSearch mapping for rank_features
  - Re-embed existing chunks (background job)
  - Validate retrieval quality (Recall@10 stable)

- [ ] **11.2.3** Deploy to production:
  - Deploy vLLM service to GPU nodes
  - Deploy updated gateway and orchestration
  - Monitor metrics for 24 hours
  - Validate embedding throughput ≥1000 emb/sec
  - Validate GPU utilization 60-80% (healthy range)
  - Validate no CPU fallbacks occurred

### 11.3 Post-Deployment Validation

- [ ] **11.3.1** Monitor for 48 hours:
  - Embedding throughput: ≥1000 emb/sec ✅
  - GPU utilization: 60-80% ✅
  - FAISS search latency: P95 <50ms ✅
  - OpenSearch sparse search: P95 <200ms ✅
  - Retrieval quality: Recall@10 stable or improved ✅
  - Zero CPU fallbacks ✅

- [ ] **11.3.2** Performance report:
  - Document: Throughput improvements (5x vs legacy)
  - Document: Latency improvements (FAISS <50ms vs ad-hoc 200ms)
  - Document: GPU utilization (healthy 60-80% range)
  - Document: Codebase reduction (25%, 130 lines removed)

- [ ] **11.3.3** Lessons learned:
  - Document: What worked well
  - Document: What was challenging
  - Document: Recommendations for future improvements

---

## Summary

**Total Tasks**: 300+ tasks across 15 work streams (UPDATED from Gap Analysis)

| Work Stream | Tasks | Duration | Status |
|-------------|-------|----------|--------|
| 1. Legacy Decommissioning | 56 | Week 1-2 | Core requirement |
| 2. Foundation | 10 | Week 1 | Core requirement |
| 3. vLLM Dense Embedding | 15 | Week 1-2 | **LIBRARY**: vllm>=0.3.0, openai>=1.0.0 |
| 4. Pyserini Sparse Embedding | 12 | Week 1-2 | **LIBRARY**: pyserini>=0.22.0 |
| 5. Multi-Namespace Registry | 15 | Week 2 | **LIBRARY**: pydantic-settings |
| 6. FAISS Storage Integration | 12 | Week 2-3 | **LIBRARY**: faiss-gpu>=1.7.4 |
| 7. Testing | 81 | Week 3-4 | Core requirement |
| 8. Performance Optimization | 10 | Week 3 | Core requirement |
| 9. Monitoring & Observability | 10 | Week 3 | **Enhanced from Gap Analysis** |
| 9A. API Integration | 20 | Week 3 | **NEW from Gap Analysis** |
| 9B. Security & Multi-Tenancy | 14 | Week 3 | **NEW from Gap Analysis** |
| 9C. Configuration Management | 16 | Week 3 | **NEW from Gap Analysis** |
| 9D. Rollback Procedures | 10 | Week 3 | **NEW from Gap Analysis** |
| 10. Documentation | 10 | Week 4 | Core requirement |
| 11. Production Deployment | 15 | Week 5-6 | Core requirement |

**Key Libraries** (Explicit Emphasis):

- **vllm>=0.3.0** - OpenAI-compatible serving for Qwen3-Embedding-8B
- **pyserini>=0.22.0** - SPLADE-v3 wrapper with document-side expansion
- **faiss-gpu>=1.7.4** - GPU-accelerated dense vector search (HNSW index)
- **openai>=1.0.0** - Client library for vLLM OpenAI-compatible API
- **transformers>=4.38.0** - Qwen3 tokenizer for token budget validation
- **pydantic-settings** - Configuration management (vLLM, namespace registry, Pyserini)

**Timeline**: 6 weeks total

- **Week 1-2**: Build new architecture + atomic deletions (vLLM, Pyserini, namespace registry)
- **Week 3-4**: Integration testing + gap analysis items (API, security, config, rollback)
- **Week 5-6**: Production deployment + monitoring

**Gap Analysis Additions** (+60 tasks):

- 20 tasks: API Integration (REST/GraphQL/gRPC namespace management)
- 14 tasks: Security & Multi-Tenancy (tenant isolation, namespace access control)
- 16 tasks: Configuration Management (vLLM, namespace registry, Pyserini YAML configs)
- 10 tasks: Rollback Procedures (automated triggers, RTO 5-20 min, post-incident analysis)

**Breaking Changes**: 4 (API signature, GPU fail-fast, FAISS primary, rank_features)
**Code Reduction**: 25% (530 → 400 lines)
**GPU-Only Enforcement**: 100% (zero CPU fallbacks)
**Legacy Decommissioning**: Comprehensive (56 tasks, atomic deletions, delegation validation)
