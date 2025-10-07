Below is an **expanded, execution‑grade embedding subsystem design** that folds in every methodology and toolchain covered by your scaffold **plus** the attached research report. I keep it **English‑first**, keep **rerankers separate** (different port), and express everything as **pluggable adapters + per‑model namespaces** so agents can wire, test, and swap models without touching core logic.

Where I restate facts (e.g., why/when to use dense vs. late‑interaction; field‑selection; LLM‑based embeddings; SPLADE/neural‑sparse; SimLM/RetroMAE/GTR; HyDE/DSI as augmentations), they are traced to the research PDF you shared.

---

## A. What changes versus your current scaffold

1. **Final universal interface** spans **single‑vector**, **multi‑vector (late‑interaction)**, **learned‑sparse**, and **neural‑sparse**; each instance is identified by a **namespace** (e.g., `dense.bge.1024.v1`).
2. **Adapter inventory** now includes: Sentence‑Transformers/FlagEmbedding, HF **TEI** server, **vLLM** (Qwen‑3), **ColBERT‑v2 indexer**, **SPLADE/uniCOIL/DeepImpact** (Pyserini), **OpenSearch neural‑sparse** encoders, and “experimental” tracks (SimLM/RetroMAE/GTR; DSI adapter skeleton).
3. **Per‑namespace storage mapping**: dense→Qdrant/FAISS/Milvus; learned‑sparse→OpenSearch `rank_features`; neural‑sparse→OpenSearch **neural** fields; ColBERT→its FAISS shards with metadata pointers.
4. **Strict dimension/version governance** (auto‑introspection + refusal on mismatch) and **English‑only presets**.
5. **Eval harness**: side‑by‑side indexing + Recall@K/nDCG@K leaderboard by namespace; regression guards baked into CI.

---

## B. Universal embedding interface (authoritative)

> Covers all first‑stage retrieval encoders only. Cross‑encoders (rerankers) live in `med/rerank/*` and are configured independently (per your instruction to keep them separate).

```python
# med/embeddings/ports.py
from typing import Iterable, Protocol, Literal, Sequence
from pydantic import BaseModel

EmbeddingKind = Literal["single_vector", "multi_vector", "sparse", "neural_sparse"]

class EmbeddingRecord(BaseModel):
    model_id: str                # HF repo or model slug
    namespace: str               # e.g., "dense.bge.1024.v1"
    kind: EmbeddingKind
    dim: int | None
    vectors: list[list[float]] | None     # 1 x d for single, N x d for multi-vector
    terms: dict[str, float] | None        # sparse/neural-sparse term→weight
    pooling: str | None                   # "mean", "cls", "late_interaction", …
    normalized: bool = True
    meta: dict = {}

class BaseEmbedder(Protocol):
    model_id: str
    kind: EmbeddingKind
    dim: int | None
    def embed_documents(self, texts: Iterable[str]) -> Sequence[EmbeddingRecord]: ...
    def embed_queries(self, texts: Iterable[str]) -> Sequence[EmbeddingRecord]: ...
```

**Why this shape?**
– *Dense bi‑encoders (single‑vector)* and *LLM‑based embedders* map “text → 1×D vector.” The report highlights Transformers/bi‑encoders (BERT/SBERT, E5/GTE), and LLM‑derived embedders like **Qwen‑3** (up to **4096‑D**)—we introspect and lock dims per namespace to eliminate schema drift.
– *Late‑interaction (ColBERT)* maps “text → many token vectors” (multi‑vector). Use `pooling="late_interaction"`; we store a pointer to ColBERT’s FAISS shards.
– *Learned‑sparse (SPLADE/uniCOIL/DeepImpact)* and *neural‑sparse* return **term→weight** maps for inverted indexes.

---

## C. Registry, config, and storage routing

### C1) YAML (embedding excerpt)

```yaml
embeddings:
  # The namespaces we will search/fuse at query time (order defines dense default)
  active_namespaces:
    - dense.bge.1024.v1
    - sparse.splade.v3.v1
    - multi.colbertv2.128.v1

  providers:
    - driver: sentence_transformers
      model_id: BAAI/bge-large-en-v1.5
      namespace: dense.bge.1024.v1
      dim: 1024
      pooling: mean
      normalize: true
      query_prefix: "query: "        # enforce E5-style usage when desired
      passage_prefix: "passage: "

    - driver: splade_torch
      model_id: naver/splade-v3-lexical
      namespace: sparse.splade.v3.v1
      topk_terms: 400

    - driver: colbert_indexer
      model_id: colbert-ir/colbertv2.0
      namespace: multi.colbertv2.128.v1
      max_doc_tokens: 200

    - driver: tei_http
      model_id: jinaai/jina-embeddings-v3
      endpoint: http://tei:8080
      namespace: dense.jina.1024.v1
      normalize: true

    - driver: openai_compat  # served by vLLM, not OpenAI cloud
      model_id: Qwen/Qwen3-Embedding-8B
      endpoint: http://vllm:8000/v1/embeddings
      namespace: dense.qwen3.4096.v1  # actual dim auto-validated
      normalize: true
```

### C2) Storage routing (universal)

| `kind`          | Default store & mapping                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| `single_vector` | **Qdrant** collection per namespace (`size=dim`, `cosine`), snapshots, optional scalar/PQ quantization.  |
| `multi_vector`  | **ColBERT** FAISS shards managed by the internal indexer; store a `colbert://index=.../docid` pointer in chunk payload.  |
| `sparse`        | **OpenSearch** `rank_features` field for SPLADE/uniCOIL/DeepImpact weights.                              |
| `neural_sparse` | **OpenSearch** neural‑sparse fields + `neural` query type (OpenSearch ML encoder hosting or remote TEI). |

The report emphasizes hybrid stacks (dense + sparse). We preserve that as first‑class: you can enable any mix of namespaces and let fusion combine them.

---

## D. Adapter inventory (what we ship)

```
med/embeddings/
  ports.py                        # BaseEmbedder, EmbeddingRecord
  st_embedder.py                  # SentenceTransformersEmbedder (BGE/E5/GTE/SPECTER/SapBERT/SimCSE/RetroMAE/SimLM/GTR)
  tei_embedder.py                 # TEIHTTPEmbedder (HF Text-Embeddings-Inference: Jina v3, E5, GTE, etc.)
  openai_compat_embedder.py       # Qwen-3 via vLLM (OpenAI-compatible)
  colbert_indexer.py              # ColbertIndexerEmbedder (multi-vector)
  splade_doc.py                   # SpladeDocExpander (doc-side term weights)
  pyserini_sparse.py              # uniCOIL / DeepImpact / TILDE exporters to OpenSearch rank_features
  neural_sparse_os.py             # OpenSearchNeuralSparseEmbedder (encoder-hosted fields)
  langchain_adapter.py            # wrap any langchain.embeddings.* as BaseEmbedder
  llamaindex_adapter.py           # wrap LlamaIndex embedding classes
  haystack_adapter.py             # wrap Haystack embedders for experiments
  dsi_searcher.py                 # (optional research) Differentiable Search Index adapter skeleton
```

**Rationale & sourcing from the report**

* **Dense/bi‑encoders** (BERT/SBERT/E5/GTE; scientific/biomed SPECTER, SapBERT), trained via contrastive/ranking losses, are your default English retrievers.
* **LLM‑based embeddings** (e.g., **Qwen‑3‑Embedding**) give long‑context, high‑dimensional vectors (e.g., **4096‑D**), but require GPU serving—vLLM path is pre‑wired.
* **Late-interaction** (**ColBERT-v2**) is multi-vector with MaxSim scoring—excellent for precise biomedical terms; we ship an internal indexer backed by ColBERT.
* **Learned sparse** (**SPLADE**, **uniCOIL/DeepImpact**) and **neural‑sparse** (OpenSearch) bring powerful lexical alignment at inverted‑index speeds; both mapped in OS.
* **Experimental dense** (**SimLM**, **RetroMAE**, **GTR**): each is a plug‑in of `SentenceTransformersEmbedder` or TEI; we enforce correct dims and pooling.
* **DSI** is supported as a *separate Searcher* (optional research), because it bypasses vector indexes.

---

## E. Per‑family integration details (actionable)

> The items below are **what agents implement and how**; prompts/pooling/dims follow the report and canonical model cards.

### E1) Dense bi‑encoders (English‑first defaults)

**Adapters:** `SentenceTransformersEmbedder` (in‑proc) and/or `TEIHTTPEmbedder` (server).
**Pooling:** `mean`; **Normalization:** L2 (cosine KNN).
**Prompts:**

* **E5** requires `query: …` / `passage: …` prefixes—enforce via config flags to prevent silent regressions.
* **BGE/GTE/Jina/SPECTER/SapBERT**: plain text is sufficient; keep toggle for E5‑style prefixes if you want consistency.

**Namespace examples and recommended uses (English):**

* `dense.bge.1024.v1` — **BGE‑large‑en**: strong general English retrieval.
* `dense.e5.1024.v2` — **E5‑large‑v2**: reliable baseline; **prefixes required**.
* `dense.gte.1024.v1` — **GTE‑large**: good accuracy/latency trade‑off.
* `dense.specter.768.v1` — SPECTER: paper‑level embedding (scholarly biomedical, abstracts).
* `dense.sapbert.768.v1` — SapBERT: UMLS/RxNorm synonym alignment; good for code matching.

**Indexing:** Qdrant collections `size=dim`, `distance=cosine`; payload: `{chunk_id, doc_id, section, year}`.

### E2) LLM‑based embeddings (Qwen‑3 family)

**Adapter:** `openai_compat_embedder.py` targeting your **vLLM** server.
**Behavior:** introspect output `dim` at startup; refuse to upsert to a mismatched collection; **GPU required**; batched requests (128–256). The report notes Qwen‑3 embedders (0.6B–8B) achieve SOTA but demand GPU; our service fail‑fasts on missing CUDA.

**Namespace:** `dense.qwen3.4096.v1` (or `.1024.` if you choose a smaller checkpoint).
**Indexing:** Qdrant with optional **scalar quantization** to reduce 4k‑D footprint; keep snapshots on.

### E2a) Framework adapters (LangChain, Haystack, LlamaIndex)

Framework-backed embedders are normalised through the shared
`DelegatedFrameworkAdapter`. Each adapter simply declares the delegate
class path and a fallback order of delegate methods (e.g.,
`embed_documents`, `embed_query`, `get_text_embedding`). The helper handles
delegate loading, L2 normalisation, offset extraction, and construction of
`EmbeddingRecord` objects so framework integrations automatically inherit
namespace validation and metadata propagation.

### E3) Late‑interaction (ColBERT‑v2)

**Adapter:** `colbert_indexer.py`.
**Build:** tokenize chunks (cap ~200 tokens), run ColBERT to produce token vectors, build FAISS shards via the internal indexer.
**Query:** embed query tokens; MaxSim across doc token vectors; return scored doc/chunk IDs.
**Storage:** ColBERT’s native index path stored in payload (`colbert://…`) for traceability; fuse rankings with dense/sparse before rerank.

### E4) Learned‑sparse (SPLADE / uniCOIL / DeepImpact / TILDE)

**SPLADE (doc‑side) pipeline:** `splade_doc.py` (Torch GPU) → compute **top‑K** term weights per chunk → upsert into OpenSearch `rank_features` (field e.g., `splade_terms`). **Optional query‑side SPLADE** encoder at runtime.

**Alternatives:** `pyserini_sparse.py` to produce **uniCOIL/DeepImpact** weights and bulk‑update OS docs— kept as additional sparse namespaces to A/B against SPLADE.

### E5) Neural‑sparse (OpenSearch)

**Adapter:** `neural_sparse_os.py`.
**Hosting:** OpenSearch ML or external TEI endpoint.
**Index:** define neural‑sparse field(s); the adapter writes encoded postings; queries use `neural` type. This is complementary to SPLADE and can be fused.

### E6) Advanced dense (SimLM, RetroMAE, GTR)

**Adapter:** `SentenceTransformersEmbedder` or `TEIHTTPEmbedder`.
**Notes from report:**

* **SimLM**: pre‑training with a representation bottleneck; very strong 1‑vector retriever.
* **RetroMAE**: MAE tuned for retrieval; robust first‑stage retriever.
* **GTR** (T5‑based): scalable English retriever (Base—XXL).
  All integrate identically to BGE/E5; your A/B harness will validate on biomedical tasks.

### E7) DSI (research track, optional)

**Adapter:** `dsi_searcher.py` (implements a *Searcher*, not an *Embedder*).
**When enabled:** the router can retrieve doc IDs straight from a seq2seq model trained to emit IDs—kept as a toggle for static corpora; not recommended for continuously updated indexes.

---

## F. Field selection & chunk shaping (embedder‑aware)

The report recommends feeding **section headings/title with paragraph text** to embedders when helpful (e.g., pair title with body in models that support pairs) to improve English retrieval quality. We expose two per‑embedder knobs: `include_title_path: true` and `join_with: " | "`; for pair‑aware models we support `(text, text_pair)` mode.

**Why:** Dense models benefit from localized context; headings sharpen intent; OpenSearch mappings also boost headings in lexical search—keeping dense and sparse in sync.

---

## G. Bootstrap & guards (dimension, GPU, mappings)

1. **Per‑namespace Qdrant collections** from config:

   * `size` = `dim`, `distance=cosine`, HNSW `{m, ef_construct}`, **quantization** optional.
   * Create payload indexes on `{doc_id, section, year}`.
   * **Refuse run** if model‑reported `dim` != collection `size`.

2. **OpenSearch indices**:

   * `rank_features` field for SPLADE/uniCOIL/DeepImpact (e.g., `splade_terms`).
   * Optional **neural‑sparse** fields + analyzer.
   * Analyzer uses your biomedical synonym graph (unchanged).

3. **GPU enforcement**:

   * Embedding/SPLADE services run only with CUDA present; otherwise **exit fast** to avoid CPU thrash (explicitly recommended in the report).

---

## H. Orchestration & performance

* **Serving:**
  – Small/medium models: in‑process `sentence-transformers` (GPU).
  – Large (Qwen‑3, big GTR): **vLLM** or **TEI server**; batch 128–256; async client with 429 backoff.
* **Throughput goals:** thousands of chunks/sec across workers; embed stage is isolated and horizontally scalable.
* **Caching:** keyed on `(namespace, text_hash)` for queries; embedding replay safety.
* **Provenance:** every upsert stores `model_id`, `namespace`, and `version` for surgical reindex.

---

## I. Evaluation harness (English‑only) & CI gates

* **Corpora:** PMC (English), DailyMed SPL, CT.gov (English).
* **Tasks:** endpoint queries, eligibility matches, AE lookups.
* **Metrics:** Recall@{10,20}, nDCG@{10}, latency; **per‑namespace dashboards**.
* **A/B:** index in parallel with new namespace; compare before swapping; store results under `/eval/<date>/<namespace>.json`.
* **CI guards:** fail build if top‑line Recall@20 degrades >X% vs baseline; dimension check unit tests; GPU‑presence self‑test.

---

## J. Implementation tasks for agents (acceptance criteria)

**1) Core & config**

* [ ] Implement `EmbeddingRecord`/`BaseEmbedder` (above) and the **namespace router**.
* [ ] Extend `RetrievalSettings` with `embeddings.active_namespaces` and `providers[]`.
* [ ] Add **dimension introspection** and refusal logic in each adapter.

**2) Dense adapters**

* [ ] `SentenceTransformersEmbedder` (BGE/E5/GTE/SPECTER/SapBERT/SimCSE/RetroMAE/SimLM/GTR).

  * Accept `pooling`, `normalize`, `query_prefix`/`passage_prefix`.
* [ ] `TEIHTTPEmbedder` (HF TEI).
* [ ] `OpenAICompatEmbedder` (Qwen‑3 via vLLM).

**3) Multi‑vector**

* [ ] `ColbertRagatouilleEmbedder` (index builder + query path); store index URI pointer and expose scores for fusion.

**4) Sparse paths**

* [ ] `SpladeDocExpander` (GPU); write `rank_features` to OS.
* [ ] `pyserini_sparse.py` (uniCOIL/DeepImpact exporters).
* [ ] `OpenSearchNeuralSparseEmbedder` for neural‑sparse.

**5) Storage bootstrap**

* [ ] Qdrant per‑namespace collections; scalar/PQ quantization toggle; snapshot cron.
* [ ] OS mappings: `rank_features` + neural‑sparse fields; synonym analyzer.

**6) Field selection**

* [ ] Add `include_title_path` + `(text_pair)` support in embedders, per report guidance on pairing heading/title with body.

**7) Orchestration**

* [ ] Compose services for TEI and vLLM; **GPU required**; embedding worker concurrency and back‑pressure.

**8) Eval**

* [ ] `eval/runner.py`: fixed query sets; leaderboards per namespace; fusion weight tuner.

**Done criteria:** query `/retrieve` with `active_namespaces=[dense.bge.1024.v1, sparse.splade.v3.v1, multi.colbertv2.128.v1]` returns fused results with per‑component scores; CI proves no dim mismatches; eval shows ≥ baseline Recall@20.

---

## K. Why this design (grounded in your report)

* **Transformer bi‑encoders** (SBERT/E5/GTE) give fast, strong English retrieval; training via contrastive objectives is now standard—use as default dense.
* **LLM‑derived embeddings** (Qwen‑3) exploit very large context/knowledge (long inputs; high‑dim vectors) but must run on GPU—our vLLM adapter and fail‑fast guards match the report’s ops advice.
* **ColBERT‑v2** (multi‑vector) helps phrase‑ and token‑level precision—critical for biomedical terms; we integrate without disturbing the rest of the pipeline.
* **Learned sparse + neural‑sparse** are first‑class to capture domain terms efficiently (OpenSearch `rank_features` and neural fields).
* **Field selection** (prepend headings/titles) and **strict schema validation** follow the report’s embedding I/O section.
* **Eval & A/B** is mandatory for safe swaps; the report describes side‑by‑side indexing with provenance for traceability.

---

## L. Appendices

### L1) Off‑the‑shelf tools we explicitly support (English‑first)

* **Libraries**: `sentence-transformers`, `FlagEmbedding` (BGE), `transformers`, `colbert-ai` (ColBERT), `pyserini` (uniCOIL/DeepImpact), OpenSearch ML client.
* **Servers**: **HF Text‑Embeddings‑Inference (TEI)**, **vLLM**.
* **Stores**: **Qdrant** (dense), **FAISS/Milvus** (optional dense), **OpenSearch** (BM25 + SPLADE + neural‑sparse).
* **Framework adapters for experiments**: LangChain, LlamaIndex, Haystack—wrapped so they return our `EmbeddingRecord` (no lock‑in).

### L2) Representative model catalogue & dims (use any subset)

| Namespace                | Model (example)                   | Dim  | Notes (English)                            |
| ------------------------ | --------------------------------- | ---- | ------------------------------------------ |
| `dense.bge.1024.v1`      | BAAI/bge-large-en-v1.5            | 1024 | Strong default; mean pool + L2 norm.       |
| `dense.e5.1024.v2`       | intfloat/e5-large-v2              | 1024 | **Requires** `query:`/`passage:` prefixes. |
| `dense.gte.1024.v1`      | Alibaba-NLP/gte-large             | 1024 | Simple prompts; solid English baseline.    |
| `dense.jina.1024.v1`     | jinaai/jina-embeddings-v3         | 1024 | Great with TEI server.                     |
| `dense.specter.768.v1`   | allenai/specter                   | 768  | Scholarly/biomed paper‑level.              |
| `dense.sapbert.768.v1`   | cambridgeltl/SapBERT              | 768  | Ontology/code synonym focus.               |
| `dense.simlm.768.v1`     | intfloat/simlm-base-msmarco…      | 768  | Bottlenecked LM pre‑train for retrieval.   |
| `dense.retromae.768.v1`  | RetroMAE‑based checkpoint         | 768  | MAE pre‑trained retriever.                 |
| `dense.gtr.768.v1`       | sentence-transformers/gtr-t5-base | 768  | T5 dual encoder; English generalization.   |
| `dense.qwen3.4096.v1`    | Qwen/Qwen3-Embedding-8B           | 4096 | LLM‑based; long inputs; GPU via vLLM.      |
| `multi.colbertv2.128.v1` | colbert-ir/colbertv2.0            | 128× | Token‑level vectors; MaxSim; FAISS shards. |
| `sparse.splade.v3.v1`    | naver/splade‑v3‑lexical           | ~V   | Term→weight map; OS rank_features.         |
| `sparse.unicoil.v1`      | uniCOIL (pyserini pipeline)       | ~V   | Term→weight map; OS rank_features.         |
| `neural.os.v1`           | OpenSearch neural‑sparse encoder  | ~V   | OS neural fields + neural query.           |

(*V = vocab size; stored sparsely.*) All of these are OSS and run locally; commercial APIs are intentionally excluded to keep data in‑house and complexity down.

---

### L3) Two key code contracts (ready to implement)

**1) Adapter factory**

```python
# med/embeddings/factory.py
from .st_embedder import SentenceTransformersEmbedder
from .tei_embedder import TEIHTTPEmbedder
from .openai_compat_embedder import OpenAICompatEmbedder
from .colbert_indexer import ColbertIndexerEmbedder
from .splade_doc import SpladeDocExpander
from .pyserini_sparse import PyseriniSparseExporter
from .neural_sparse_os import OpenSearchNeuralSparseEmbedder

REGISTRY = {
  "sentence_transformers": SentenceTransformersEmbedder,
  "tei_http": TEIHTTPEmbedder,
  "openai_compat": OpenAICompatEmbedder,
  "colbert_indexer": ColbertIndexerEmbedder,
  "splade_torch": SpladeDocExpander,
  "pyserini_sparse": PyseriniSparseExporter,
  "neural_sparse_os": OpenSearchNeuralSparseEmbedder,
}
```

**2) Dimension guard**

```python
def check_dim_or_raise(adapter_dim: int|None, namespace: str, store_dim: int|None):
    if adapter_dim is None and store_dim is None: return
    if adapter_dim != store_dim:
        raise RuntimeError(f"Dim mismatch for {namespace}: model={adapter_dim}, store={store_dim}")
```

The report stresses **strict validation** of vector shapes and **fail‑fast** GPU checks; both are enforced here.

---
