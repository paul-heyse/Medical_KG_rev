# Embedding Adapter Catalog

The universal embedding system ships with the following adapter families.

| Adapter | Kind | Models | Parameters | Primary Use Cases |
| ------- | ---- | ------ | ---------- | ----------------- |
| SentenceTransformersEmbedder | single_vector | BGE, E5, GTE, SPECTER, SapBERT | `batch_size`, `normalize`, `prefixes`, `onnx` | General dense retrieval, scientific search, biomedical entity linking |
| TEIHTTPEmbedder | single_vector | Jina v3, Hugging Face TEI hosted models | `endpoint`, `headers`, `timeout` | Offloading inference to TEI servers |
| OpenAICompatEmbedder | single_vector | Qwen-3, vLLM-hosted OpenAI compatible models | `endpoint`, `api_key`, `model_id` | LLM-based embeddings served through OpenAI-compatible APIs |
| ColbertIndexerEmbedder | multi_vector | ColBERT-v2 | `max_doc_tokens`, `shards`, `shard_capacity`, `qdrant_collection` | Late interaction retrieval with FAISS shards or Qdrant |
| SPLADEDocEmbedder / SPLADEQueryEmbedder | sparse | SPLADE v3 | `top_k`, `normalization` | Learned sparse document and query expansion |
| PyseriniSparseEmbedder | sparse | uniCOIL, DeepImpact, TILDE | `weighting`, `normalization` | BM25-style sparse encoders with learned weights |
| OpenSearchNeuralSparseEmbedder | neural_sparse | OpenSearch ML Plugin models | `field`, `ml_model_id`, `external_endpoint` | Neural sparse retrieval with OpenSearch neural fields |
| LangChainEmbedderAdapter | single_vector | LangChain integrations | `class_path`, `init`, `include_offsets` | Bridging LangChain embeddings into the universal pipeline |
| LlamaIndexEmbedderAdapter | single_vector | LlamaIndex integrations | `class_path`, `init`, `include_offsets` | Integrating LlamaIndex embeddings |
| HaystackEmbedderAdapter | single_vector | Haystack embedders | `class_path`, `init`, `include_offsets` | Porting Haystack embedding components |

## Configuration Examples

```yaml
embeddings:
  active_namespaces:
    - single_vector.bge_small_en.384.v1
    - sparse.splade.400.v1
  providers:
    - name: bge-small
      provider: sentence-transformers
      kind: single_vector
      namespace: single_vector.bge_small_en.384.v1
      model_id: BAAI/bge-small-en-v1.5
      batch_size: 32
      normalize: true
      parameters:
        onnx: true
        progress_interval: 64
    - name: splade
      provider: splade-doc
      kind: sparse
      namespace: sparse.splade.400.v1
      model_id: naver/splade-v3-lexical
      parameters:
        top_k: 400
        normalization: l2
```

## Evaluation Harness Usage

```python
from Medical_KG_rev.eval import EmbeddingEvaluator, EvaluationDataset

# Build dataset
beir = EvaluationDataset(
    name="toy",
    queries={"q1": ["aspirin safety"]},
    relevant={"q1": {"doc-42"}},
)

# Define retrieval callback
from Medical_KG_rev.services.retrieval.retrieval_service import RetrievalService
retrieval = RetrievalService(...)

 def retrieve(namespace: str, text: str, k: int):
     return retrieval._vector_store_search(text, k, context)

# Run evaluation
metrics = EmbeddingEvaluator(beir, retrieve).evaluate("single_vector.bge_small_en.384.v1")
```

## Deployment Notes

### Text-Embeddings-Inference Server

1. Launch the server with Docker:
   ```bash
   docker run --rm -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest \
     --model-id jinaai/jina-embeddings-v3
   ```
2. Update the provider block to point to `http://localhost:8080` and provide any required headers.

### vLLM Embedding Endpoint

Start a vLLM instance using the OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Embedding-8B \
  --port 8000
```

Configure the `OpenAICompatEmbedder` with the endpoint `http://localhost:8000/v1/embeddings` and include any bearer tokens via
`parameters.headers.Authorization`.

### Model Download Helper

Use the bundled script to pre-download frequently used models:

```bash
python scripts/download_models.py --models BAAI/bge-small-en-v1.5 naver/splade-v3-lexical
```

The script stores models under the project cache directory so that CI environments and air-gapped deployments can reuse them.
