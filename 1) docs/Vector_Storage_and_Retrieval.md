Below I **merge and extend** your prior catalogue with **every tool and methodology** called out in the attached research report and our earlier designs—**text‑only**, English‑first, open‑source, and fully local. I keep the same **ports‑and‑adapters** shape you’re already using and add **advanced vector compression** and **experimental designs** with concrete classes, YAML keys, and index parameters so your AI agents can implement immediately.

> Citations labeled like this refer to the attached research report *Local Vector Retrieval and Reranking Systems for Biomedical Document Pipelines*. I place them exactly where the finding is used.

---

## 0) What’s new vs the last draft (at a glance)

* **New backends you can run locally**: Weaviate (hybrid HNSW + BM25F), OpenSearch k‑NN with **Faiss engine** (IVF, PQ), **DiskANN** (SSD‑resident ANN), and **Qdrant multivector** patterns (store token vectors for late interaction).
* **Advanced vector compression**: PQ/OPQ, scalar (int8/fp16), **binary quantization (BQ)** in Qdrant, and two‑stage “compressed‑search then re‑order with floats.”
* **GPU paths**: Qdrant **GPU‑accelerated indexing**, Milvus **GPU IVF/PQ** and **GPU_CAGRA**, FAISS GPU brute‑force baselines.
* **Reranking**: explicit lexical rerank (BM25/BM25F), RRF fusion, cross‑encoders (BGE, MiniLM, MonoT5), **ColBERTv2 as re‑ranker**; all kept **separate** under `RerankerPort`.

No multimodal tooling is included (per your instruction).

---

## 1) Universal adapter shape (unchanged, extended with compression)

### 1.1 Ports (dense vector stores, sparse stores, rerankers)

```python
# med/ports/vector_store.py
from typing import Any, Iterable, Protocol, Sequence
from pydantic import BaseModel

class CompressionPolicy(BaseModel):
    kind: str | None = None        # None | "scalar_int8" | "fp16" | "pq" | "opq_pq" | "binary"
    pq_m: int | None = None        # subvector count (PQ)
    pq_nbits: int | None = None    # codebook bits (4 or 8 typical)
    opq_m: int | None = None       # OPQ rotation blocks (if using OPQ)
    notes: dict[str, Any] = {}

class IndexParams(BaseModel):
    kind: str                       # "hnsw" | "ivf_flat" | "ivf_pq" | "flat" | "diskann" | "knn_vector"
    metric: str                     # "cosine" | "ip" | "l2"
    dim: int
    # HNSW
    m: int | None = None
    ef_construct: int | None = None
    ef_search: int | None = None
    # IVF
    nlist: int | None = None
    nprobe: int | None = None
    # DiskANN
    disk: dict | None = None       # paths, queue_depth, mem_limits
    # OpenSearch extras
    engine: str | None = None      # "lucene" | "faiss"
    encoder: dict | None = None    # {"type":"pq"|"sq"|"none", ...}
    # Compression
    compression: CompressionPolicy | None = None

class VectorStorePort(Protocol):
    def create_or_update_collection(self, namespace: str, params: IndexParams) -> None: ...
    def upsert(self, namespace: str, ids: Sequence[str], vectors: Sequence[list[float]], payloads: Sequence[dict]) -> None: ...
    def knn(self, namespace: str, query_vec: list[float], top_k: int, filters: dict | None = None) -> list[dict]: ...
    def delete(self, namespace: str, ids: Sequence[str]) -> None: ...
```

*Every backend listed below implements this interface; all **compression knobs live in `IndexParams.compression`** so a single config schema controls Qdrant/FAISS/Milvus/OpenSearch the same way.*

### 1.2 YAML: per‑namespace, per‑backend, with compression

```yaml
vector_store:
  driver: qdrant | faiss | milvus | opensearch_knn | pgvector | weaviate | diskann | hnswlib | annoy | nmslib | scann | vespa | lancedb | duckdb_vss
  collections:
    dense.bge.1024.v1:
      kind: hnsw
      dim: 1024
      metric: cosine
      m: 64
      ef_construct: 400
      ef_search: 128
      compression: { kind: "scalar_int8" }        # Qdrant int8 or FAISS SQ8, OS: encoder sq/fp16
    dense.qwen.1536.v1:
      kind: ivf_pq
      dim: 1536
      metric: ip
      nlist: 32768
      nprobe: 64
      compression: { kind: "opq_pq", opq_m: 64, pq_m: 64, pq_nbits: 8 }
```

---

## 2) Dense vector storage & lookup — **ALL local options** (how to wire them)

> Each entry includes **why**, **adapter file**, **key params**, **compression**, and **notes** specific to the engine. All are text‑only.

### 2.1 Qdrant (Rust HNSW; filters; **int8 & binary quantization**; **GPU indexing**) — **default**

*Adapter:* `vstore/qdrant_store.py`
*Index:* HNSW (`m`, `ef_construct`, `ef_search`) with payload filters; **on‑disk vectors + in‑RAM graph**; **named vectors** for multi‑embedding; **can disable HNSW** on fields used only in re‑rank. Qdrant supports **scalar int8** and **binary quantization (BQ)**; **GPU‑accelerated indexing** speeds up HNSW build.

**Compression**
`compression.kind: "scalar_int8"` for 4× memory reduction with small recall hit; or `"binary"` (BQ) for up to **40× speedups** when you **re‑order top‑candidates with original floats** to recover accuracy.

**Multivector / late interaction**
Use Qdrant **named vectors** to store token‑level vectors (not indexed) for ColBERT‑style rerank: index a primary vector; fetch token vectors for **MaxSim** in the reranker.

**Notes**
Strong metadata filtering; containerized single binary. Validate dimensions at upsert (your existing guard).

---

### 2.2 FAISS (C++/GPU) — Exact & ANN, maximum control

*Adapter:* `vstore/faiss_store.py`
*Index kinds:* `Flat` (exact), `IVF_FLAT`, `IVF_PQ`, `HNSW`, `OPQ+IVF_PQ`. **GPU** variants provide brute‑force or IVF search on device.

**Compression**
*Scalar:* SQ8 / fp16; *PQ/OPQ:* codebooks via `index_factory` strings (e.g., `OPQ64,IVF32768,PQ64x8`). Use **reorder**: retrieve 1000 by PQ distance then re‑score final 100 with float vectors.

**Persistence**
`faiss.write_index` per namespace; store `id→payload` in SQLite/LMDB sidecar for filters.

---

### 2.3 Milvus / Milvus‑Lite (C++/Go; **GPU IVF/PQ & CAGRA**; DiskANN option)

*Adapter:* `vstore/milvus_store.py`
*Index kinds:* `IVF_FLAT`, `IVF_PQ`, `HNSW`, **DISKANN**; **GPU** builds support **GPU_IVF_FLAT**, **GPU_IVF_PQ**, **GPU_CAGRA** graph.

**Compression**
Configure **IVF+PQ** with `nlist/nprobe`, `pq_m`, `pq_nbits`; combine with OPQ upstream if needed.

**Notes**
Hybrid vector+scalar filters; paging from disk via mmaps/WAL; “Lite” mode embeds in Python for single‑node.

---

### 2.4 OpenSearch k‑NN (Lucene HNSW) **and FAISS engine** (IVF & PQ) — one‑engine hybrid

*Adapter:* `vstore/opensearch_knn_store.py`
*Lucene engine:* `knn_vector` (HNSW) (`m`, `ef_construction`; `ef_search≈k`), integrates with BM25 and **rank_features** in the same index.
*FAISS engine:* unlocks **IVF** and **PQ** encoders (set `encoder: {type: "pq"| "sq"}`; **8‑bit codes** typical; **_train** API for centroids/codebooks).

**Compression**
*fp16 scalar* via `encoder: sq` for half‑precision; *PQ* with `m` (subvectors) and `code_size` (bits).

**Hybrid**
Same index can hold `rank_features` (SPLADE), dense `knn_vector`, and classic fields for BM25/BM25F.

---

### 2.5 Weaviate (Go; HNSW + **built‑in hybrid fusion BM25f**)

*Adapter:* `vstore/weaviate_store.py`
*Index:* HNSW (ef, M) over vectors; **hybrid search** fuses BM25 and vector similarity with a built‑in weighting scheme; GraphQL API + Python client. Good when you want schemaful classes and **one process** for dense+lexical.

---

### 2.6 Vespa (Java/C++) — hybrid with **rank profiles** & ONNX

*Adapter:* `vstore/vespa_store.py`
*Index:* HNSW + BM25; define **rank profiles** to combine cosine/IP with BM25 and ONNX rerank logic in one pipeline. Suited to custom LTR at retrieval time.

---

### 2.7 pgvector (PostgreSQL)

*Adapter:* `vstore/pgvector_store.py`
*Index kinds:* `ivfflat` (lists/probes) and **HNSW** (m, ef) in recent pgvector; use SQL filters and joins when your biomedical data lives in Postgres.

---

### 2.8 DiskANN (C++; SSD‑resident ANN)

*Adapter:* `vstore/diskann_store.py` (wrap `diskannpy`)
*Index:* Vamana graph on **NVMe**; RAM holds graph/centroids only; queries page small chunks from disk with async I/O. Use when collections exceed RAM by far.

---

### 2.9 Embedded libraries (simple, in‑process)

* **hnswlib** → `vstore/hnswlib_index.py` (HNSW; M/ef tuning)
* **NMSLIB** → `vstore/nmslib_index.py` (HNSW & more)
* **Annoy** → `vstore/annoy_index.py` (random projection trees; static)
* **ScaNN** → `vstore/scann_index.py` (partition + asymmetric hashing; strong CPU perf)

> For these, payload filters live in a sidecar KV (SQLite/LMDB). All are fully local and useful for experiments or tenant‑local indices.

---

### 2.10 Other local stores (optional)

* **LanceDB** → `vstore/lancedb_store.py` (columnar on‑disk; evolving IVF/HNSW)
* **DuckDB‑VSS / sqlite‑vss** → `vstore/duckdb_vss_store.py` / `vstore/sqlite_vss_store.py` (compact embedded SQL + vectors)
* **ChromaDB** → `vstore/chroma_store.py` (simple local store for RAG experiments)

---

## 3) Sparse, learned‑sparse, and neural‑sparse (local)

* **BM25/BM25F (OpenSearch)** → `sparse/bm25_os.py` (field boosts; biomedical analyzers).
* **SPLADE / uniCOIL / DeepImpact (doc‑side)** → `sparse/splade_doc.py` writes **`rank_features`** per doc; optional **query‑side** SPLADE encoder.
* **Neural‑sparse (OpenSearch ML)** → `sparse/neural_sparse_os.py` with **sparse encoding processor** or HF model; query uses `neural` operator.

These shapes match your existing OpenSearch mapping (doc text fields + `rank_features`), exactly as shown in the research report.

---

## 4) **Advanced vector compression & memory‑saving** (how to expose it)

> We expose all compression methods via `CompressionPolicy`. The adapter translates that into engine‑native settings.

### 4.1 Methods to support (and where)

| Method                              | Where supported                                          | How to configure                                              | Notes                                                                                                                |                                                                       |
| ----------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Scalar quantization** (int8/fp16) | Qdrant, FAISS, OpenSearch (FAISS engine), Milvus         | `compression.kind: "scalar_int8"                              | "fp16"`                                                                                                              | ~4× memory reduction; fast SIMD distances; tiny recall drop.          |
| **PQ** (m, nbits)                   | FAISS, Milvus, OpenSearch (FAISS engine)                 | `compression: {kind:"pq", pq_m:64, pq_nbits:8}`               | 8‑bit codes common; 4‑bit possible in FAISS for extreme compression. **Reorder** final candidates by float vectors.  |                                                                       |
| **OPQ + PQ**                        | FAISS (index_factory), Milvus (upstream OPQ then IVF‑PQ) | `compression: {kind:"opq_pq", opq_m:64, pq_m:64, pq_nbits:8}` | Rotation increases accuracy for same code size.                                                                      |                                                                       |
| **Binary quantization (BQ)**        | **Qdrant**                                               | `compression.kind: "binary"`                                  | Bit‑wise distance; **40×** speedups in first‑stage, then re‑rank with floats.                                        |                                                                       |
| **IVF coarse quantization**         | FAISS, Milvus, OpenSearch (FAISS engine)                 | `kind:"ivf_pq"                                                | "ivf_flat", nlist, nprobe`                                                                                           | Sub‑linear search; pick `nlist≈√N`, `nprobe` 10–64; combine with PQ.  |

**Two‑stage best practice** (exposed via a single flag):
`search.reorder_final: true` → retrieve `R` by compressed distance (PQ/BQ) and **re‑score R by original 32‑bit vectors** before returning top‑K. Qdrant BQ and FAISS IVF/PQ benefit from this pattern.

**ColBERT‑style 2‑bit**: ColBERTv2 stores token embeddings quantized (e.g., `nbits=2`) in an IVF structure to keep indexes small; capture this under the **ColBERT indexer** (see §6.3).

---

## 5) Retrieval orchestration (dense + sparse + neural‑sparse + optional late‑interaction)

*Parallel fan‑out* to the enabled namespaces (dense KNN, BM25/BM25F, SPLADE/neural‑sparse, optional ColBERT retriever), **then fuse** results.

**Fusion options**:
*Weighted linear* (default) and **RRF** (robust baseline): `score = Σ(1/(rank_i + C))`. RRF is entirely local to implement and works well to combine dense and lexical lists.

**Hybrid within one engine**: Weaviate’s built‑in hybrid (BM25 + vector); OpenSearch can include vector kNN and lexical clauses in the same DSL. We keep both as optional “single‑engine” modes.

---

## 6) Reranking stacks (kept separate; text‑only)

> All adapters implement `RerankerPort.score_pairs(batch: list[tuple[str,str]]) -> list[float]]`.

### 6.1 Lexical rerankers

* **BM25/BM25F rerank** on the candidate set (OpenSearch `terms`/`ids` filter + weighted query). Fast and improves precision for biomedical terms.

### 6.2 Cross‑encoders (English‑first)

* `rerank/ce_bge_reranker.py` — **BAAI/bge‑reranker‑v2‑m3** (FP16 GPU; ONNX int8 for CPU)
* `rerank/ce_mini_lm.py` — **MiniLM** distilled CE (fast CPU/GPU)
* `rerank/ce_monoT5.py` — **MonoT5** (slower, strong quality)
* `rerank/ce_qwen_reranker.py` — Qwen reranker (serve via vLLM if large); vLLM/DeepSpeed can help big CEs.

### 6.3 Late‑interaction reranker (ColBERTv2)

* `rerank/colbert_reranker.py` — either use the **ColBERT indexer** for fast MaxSim, or **Qdrant multivector**: fetch token vectors and compute MaxSim locally for top‑N. Good for strings/abbreviations (drugs, outcomes).

### 6.4 LTR/ONNX profiles (engine‑native)

* `rerank/ltr_ranker.py` — OpenSearch LTR (LambdaMART/XGBoost features: BM25, SPLADE, dense score, recency) or Vespa rank profiles with ONNX.

*(LLM‑style “judge” reranking is available behind a feature flag but omitted here for brevity; still text‑only when enabled.)*

---

## 7) Implementation recipes per backend (copy‑paste settings)

### 7.1 Qdrant (HNSW + int8/BQ; **GPU indexing**)

```yaml
vector_store:
  driver: qdrant
  server: { url: "http://localhost:6333" }
  collections:
    dense.bge.1024.v1:
      kind: hnsw
      dim: 1024
      metric: cosine
      m: 64
      ef_construct: 400
      ef_search: 128
      compression: { kind: "scalar_int8" }              # int8 vectors in RAM
      search:
        reorder_final: true                             # fetch original floats to re-score
    dense.bge.1024.v1.tokens:                           # optional multivector for ColBERT rerank
      kind: none                                        # store-only; no HNSW
      dim: 128
```

*Notes:* enable **on‑disk original vectors**; keep only HNSW & compressed replica in RAM; allow **GPU indexing** flag during build if available.

### 7.2 FAISS (IVF‑PQ + OPQ; GPU)

```yaml
vector_store:
  driver: faiss
  collections:
    dense.bge.1024.v1:
      kind: ivf_pq
      dim: 1024
      metric: ip
      nlist: 32768
      nprobe: 64
      compression: { kind:"opq_pq", opq_m:64, pq_m:64, pq_nbits:8 }
      search: { reorder_final: true }
      gpu: true
```

*Train* centroids/PQ codebooks per namespace; persist via `write_index`.

### 7.3 Milvus (GPU IVF‑PQ or CAGRA)

```yaml
vector_store:
  driver: milvus
  server: { uri: "http://localhost:19530" }
  collections:
    dense.bge.1024.v1:
      kind: ivf_pq
      dim: 1024
      metric: COSINE
      nlist: 32768
      nprobe: 64
      compression: { kind:"pq", pq_m:64, pq_nbits:8 }
      gpu: { enabled: true, kind: "GPU_IVF_PQ" }
```

*Use GPU_CAGRA for high‑QPS graphs when appropriate.*

### 7.4 OpenSearch (Lucene HNSW or FAISS IVF/PQ + SPLADE rank_features)

```json
PUT /chunks
{
  "settings": { "index.knn": true, "index.knn.algo_param.m": 32, "index.knn.algo_param.ef_construction": 200 },
  "mappings": {
    "properties": {
      "vector_dense_bge_1024_v1": { "type": "knn_vector", "dimension": 1024, "similarity": "cosinesimil" },
      "splade_terms": { "type": "rank_features" },
      "title_path": { "type": "text" }, "body": { "type": "text" }
    }
  }
}
```

Or FAISS engine:

```json
PUT /chunks/_settings
{ "index.knn.space_type":"cosinesimil", "index.knn.engine":"faiss" }

PUT /chunks/_mappings
{ "properties": { "dense_bge_1024": { "type":"knn_vector", "dimension":1024,
  "method": { "engine":"faiss","space_type":"cosinesimil","name":"ivf",
              "parameters":{"nlist":32768,"encoder":{"name":"pq","m":64,"code_size":8}}}}}}
```

Train with `_train` then index. Fuse with BM25 in a bool query or client‑side RRF.

### 7.5 Weaviate (hybrid)

Define class schema with vectorizer off (we pre‑embed) and enable **hybrid search** at query time; our adapter maps GraphQL to `knn + bm25` with a weight.

### 7.6 DiskANN

```yaml
vector_store:
  driver: diskann
  collections:
    dense.bge.1024.v1:
      kind: diskann
      dim: 1024
      metric: l2
      disk: { data_path: "/data/bge_1024", mem_gb: 8, pq_bytes: 0 }
```

Build with `diskannpy.build`, serve via Python wrapper; ideal when vectors >> RAM.

---

## 8) Search‑time strategy (how the Router calls these)

1. **Dense**: KNN from default dense namespace (Qdrant/FAISS/Milvus/OS/Weaviate)
2. **Lexical**: BM25/BM25F (OpenSearch)
3. **Learned‑sparse**: SPLADE or neural‑sparse (OpenSearch ML)
4. **Optional**: ColBERT retriever (use the ColBERT indexer or Qdrant multivector)
5. **Fuse**: Weighted or **RRF**; protect exact ID hits (NCT/PMID)
6. **Rerank**: cross‑encoder or ColBERT MaxSim on top‑N

This mirrors the hybrid + two‑stage pattern emphasized in the report.

---

## 9) Tuning defaults (from the report; good English‑first baselines)

* **HNSW**: `m=64, ef_construct=400–512, ef_search=128–256` for high recall.
* **IVF**: `nlist≈√N`, `nprobe=10–64`. Combine with **PQ** and **reorder_final=true**.
* **Qdrant compression**: start with **int8**; consider **BQ** only if you re‑rank by floats and can tolerate some oversampling.
* **GPU brute‑force**: for ≤1M x 384d, FAISS GPU `Flat` can beat ANN in latency—keep as a baseline.
* **Hybrid fusion**: Dense 0.35 / SPLADE 0.50 / BM25 0.15, then calibrate via eval harness; **RRF** if you want parameter‑free robustness.
* **Rerank**: BGE CE over top‑100 (batch 16–32, FP16) or MiniLM if CPU‑only; ColBERT rerank for abbreviation‑heavy queries.

---

## 10) Experimental designs (first‑class, toggleable)

1. **Qdrant BQ → float re‑rank** (fast first pass, accurate final): `compression: binary` + `search.reorder_final: true`.
2. **ColBERTv2 “multi‑vector rerank”**: store token vectors as **named vectors**, not indexed; compute MaxSim only on top‑N.
3. **Single‑engine hybrids**: Weaviate hybrid, or all‑in‑OpenSearch (BM25 + SPLADE + kNN in one index). Useful for ops simplicity.
4. **DiskANN for NVMe‑scale**: SSD‑first ANN when RAM is the bottleneck; keep same `VectorStorePort`.
5. **GPU IVF‑PQ / CAGRA**: offload query to GPU (Milvus) when you have spare GPU and very large N.
6. **FAISS OPQ+IVF‑PQ + reorder**: strongest CPU/GPU trade‑off when memory is tight but recall must remain high.

All are **text‑only**, English‑first, and plug in via the same ports.

---

## 11) Eval harness extensions (to decide best local stack)

Add 3 small test runners your agents can implement today:

* **ANN sweep**: vary (`m`, `ef_search`) for HNSW; (`nlist`, `nprobe`) for IVF; **report Recall@10/20, QPS, memory**.
* **Compression A/B**: float32 vs int8 vs PQ vs BQ with **reorder**; plot recall/latency curves.
* **Hybrid & rerank**: compare Dense only vs Dense+BM25 vs Dense+SPLADE vs all three, with and without CE/ColBERT re‑rank; use **RRF** baseline.

Store results per **namespace** and per **backend**, matching your “settings‑first” governance (prevents dim/version drift).

---

## 12) What to code now (actionable backlog & file map)

```
med/
  vstore/
    qdrant_store.py           # HNSW + int8/BQ + GPU indexing + named vectors
    faiss_store.py            # Flat/IVF/HNSW/PQ(OPQ) + GPU + reorder
    milvus_store.py           # IVF/HNSW/DiskANN + GPU_IVF_PQ/CAGRA
    opensearch_knn_store.py   # Lucene HNSW or FAISS IVF/PQ + _train + encoder
    weaviate_store.py         # HNSW + hybrid fusion
    pgvector_store.py         # IVFFlat/HNSW inside Postgres
    diskann_store.py          # SSD-resident ANN via diskannpy
    hnswlib_index.py | nmslib_index.py | annoy_index.py | scann_index.py
    vespa_store.py | lancedb_store.py | duckdb_vss_store.py | chroma_store.py
  sparse/
    bm25_os.py                # BM25/BM25F
    splade_doc.py             # doc-side SPLADE -> rank_features
    neural_sparse_os.py       # OpenSearch ML neural-sparse pipeline
  rerank/
    ce_bge_reranker.py | ce_mini_lm.py | ce_monoT5.py | ce_qwen_reranker.py
    colbert_reranker.py       # MaxSim on top-N (ColBERT indexer or Qdrant multivector)
    ltr_ranker.py             # OS-LTR or Vespa rank profile
```

*All drivers read the same `IndexParams`/`CompressionPolicy`; your existing **dimension guard** runs before upserts and queries.*

---

## 13) Ops guardrails (from the report; adopt directly)

* **GPU‑gated** embedding & SPLADE; **fail‑fast** if CUDA absent.
* **Snapshots**: Qdrant snapshots; OpenSearch snapshot repo; FAISS index files with versioned names.
* **Index updates**: HNSW supports dynamic insert; IVF **retrain** when data distribution shifts; schedule rebuilds.
* **Throughput**: HNSW is single‑thread per query—run multiple workers; consider GPU IVF/PQ when N is huge.

---

## 14) Rationale highlights (why these choices)

* **Qdrant** as the dense default: filtering, snapshots, **int8 & BQ** options, and **GPU indexing** give a high‑performance local baseline with strong ergonomics.
* **FAISS/Milvus** unlock **OPQ+IVF‑PQ** (and GPU CAGRA/IVF) for tight memory/QPS budgets.
* **OpenSearch & Weaviate** support single‑engine **hybrid** (lexical+sparse+dense) with operational simplicity.
* **DiskANN** keeps “billions of vectors on one box” a real option with NVMe.
* **RRF + CE rerank** is the robust accuracy path for biomedical English QA.

---

### Final note

Everything above is **text‑only** and designed to be **swapped by config**. Your agents can now add the drivers listed, wire `CompressionPolicy`, and extend the eval harness to **choose** among Qdrant (HNSW+int8/BQ), FAISS/Milvus (OPQ+IVF‑PQ), OpenSearch/Weaviate (single‑engine hybrids), or **DiskANN**—without touching the core. All claims and parameters marked above are grounded in the attached research report.
