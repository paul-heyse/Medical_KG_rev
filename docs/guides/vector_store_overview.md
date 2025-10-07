# Vector Storage & Retrieval Overview

This guide summarises the capabilities delivered by the vector storage subsystem.

## VectorStorePort Interface

All adapters implement the `VectorStorePort` protocol:

| Method | Description |
| --- | --- |
| `create_or_update_collection` | Ensures a namespace exists with the provided index parameters, compression policy, and optional named vectors. |
| `list_collections` | Returns namespaces visible to the tenant. |
| `upsert` | Inserts or updates a batch of `VectorRecord` instances. |
| `query` | Executes similarity search for the provided `VectorQuery`. |
| `delete` | Removes vector IDs from the namespace. |
| `create_snapshot` | Writes a point-in-time snapshot/backup for the namespace. |
| `restore_snapshot` | Rehydrates a namespace from an existing snapshot artifact. |
| `rebuild_index` | Retrains or rebuilds the underlying index implementation. |
| `check_health` | Returns backend-specific health details for namespaces. |

Each adapter validates dimensions using `NamespaceRegistry` before writing and surfaces metadata for downstream retrieval components.

## Backend Selection Guide

| Backend | When to use |
| --- | --- |
| **OpenSearchKNNStore** | Lucene HNSW and FAISS engines with hybrid lexical + vector search, `_train` support, and rank profiles. |
| **WeaviateStore** | BM25f fusion on top of the OpenSearch delegate with configurable vector weights. |
| **VespaStore** | RRF rank profiles and ONNX reranking on FAISS-backed vectors. |
| **PgvectorStore** | PostgreSQL deployments requiring IVFFLAT tuning and SQL compatibility. |
| **DiskANNStore** | SSD-optimised ANN with precomputed distance caches. |
| **HNSWLib / NMSLib / Annoy / ScaNN** | Embedded libraries for lightweight deployments. |
| **LanceDB / DuckDBVSS / Chroma** | Local development, analytics workflows, and rapid RAG prototyping. |

Each adapter advertises capabilities via `detect_backend_capabilities`, allowing configuration code to select GPUs, compression types, and hybrid features dynamically.

## Compression Policies

Compression is configured with `CompressionPolicy`:

```yaml
compression:
  kind: pq
  pq_m: 16
  pq_nbits: 8
```

Supported kinds include `none`, `int8`, `fp16`, `pq`, and `opq`. The compression manager in `services/vector_store/compression.py` validates options and integrates with evaluation utilities for A/B testing.

## YAML Configuration

`config/vector_store.yaml` uses the following structure:

```yaml
backends:
  qdrant:
    url: http://localhost:6333
tenants:
  - tenant_id: clinical
    namespaces:
      - name: dense
        driver: qdrant
        params:
          dimension: 768
          metric: cosine
          kind: hnsw
        compression:
          kind: int8
```

`migrate_vector_store_config` normalises legacy files while `detect_backend_capabilities` inspects adapters to suggest compatible drivers.

## GPU Integration

`GPUResourceManager` enforces fail-fast semantics, plans batch sizes (`plan_batches`), and records fallbacks for observability. Metrics are exposed via `vector_operation_duration_seconds`, and GPU utilisation is surfaced through `summarise_stats(get_gpu_stats())`.

## Troubleshooting

- **Dimension mismatches** – validated by `NamespaceRegistry.ensure_dimension` before upserts and queries.
- **GPU missing** – `GPUFallbackStrategy` logs `vector.gpu_fallback` events and gracefully routes to CPU.
- **Compression issues** – `compression_ab_test` benchmarks policies and `record_compression_ratio` exposes ratios for dashboards.
- **Hybrid routing** – `RetrievalService.search` accepts `embedding_kind` to select namespaces per embedding family.
- **Snapshot recovery** – use `VectorStoreService.restore_snapshot` with `overwrite=True` to rebuild namespaces from backups.
- **Health checks** – call `VectorStoreService.check_health` to surface backend readiness for observability pipelines.

Refer to `services/vector_store/evaluation.py` for parameter sweeps, latency profiles, and leaderboard generation used in continuous evaluation.

