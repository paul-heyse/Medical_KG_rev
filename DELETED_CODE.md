# Deleted Legacy Orchestration Code

| File | Reason for Removal |
| ---- | ------------------ |
| `src/Medical_KG_rev/services/retrieval/indexing_service.py` | Replaced by the Dagster/Haystack index stage. Dual write logic now lives in `Medical_KG_rev.orchestration.haystack.components.HaystackIndexWriter`. |
| `src/Medical_KG_rev/services/embedding/service.py` | Legacy embedding worker with custom batching and vector-store side effects. Superseded by the stage-backed `EmbeddingWorker` that delegates to the Dagster embed stage. |
| `tests/services/embedding/test_embedding_registry.py` | Exercised legacy registry behaviour tied to the removed worker implementation. |
| `tests/services/embedding/test_embedding_vector_store.py` | Relied on the deleted embedding worker side effects. |
| `tests/services/retrieval/test_indexing_service.py` | Validated the removed indexing service. |
| `tests/services/test_retrieval_query_cache.py` | Depended on the bespoke embedding worker and indexing service cache semantics. |
| `tests/embeddings/test_core.py` | Covered the legacy embedding worker CLI and batch execution paths. |

Removal of these modules keeps the codebase aligned with the Dagster-driven
orchestration design and avoids maintaining unused compatibility shims.

