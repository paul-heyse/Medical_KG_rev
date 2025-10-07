# Chunking API Overview

The chunking module exposes a high-level service and factory used by ingestion and retrieval
pipelines.

## Service Interface

```python
from Medical_KG_rev.chunking import ChunkingService, ChunkingOptions

service = ChunkingService()
chunks = service.chunk_text(
    tenant_id="tenant-123",
    document_id="doc-456",
    text="Introduction...",
    options=ChunkingOptions(strategy="semantic_splitter", granularity="paragraph"),
)
```

The `ChunkingService` resolves multi-granularity profiles from `config/chunking.yaml` and preserves
full provenance (`chunk.meta['block_ids']`, `chunk.start_char`, `chunk.end_char`).

## Ingestion Integration

`IngestionService` wraps `ChunkingService` to manage chunk storage, latency telemetry, and
profile detection.

```python
from Medical_KG_rev.services.ingestion import IngestionService

service = IngestionService()
run = service.chunk_document(document, tenant_id="tenant-123", source_hint="pmc")
print(run.granularity_counts)
```

## Retrieval Integration

`RetrievalService` now exposes granularity-aware search with weighted fusion and window merging.

```python
results = retrieval_service.search("chunks", "hypertension", filters={"granularity": "paragraph"})
for result in results:
    print(result.granularity, result.text[:80])
```
