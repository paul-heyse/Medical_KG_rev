"""Service to route ingestion batches into the configured vector store namespaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from Medical_KG_rev.auth.context import SecurityContext

from ..vector_store.models import UpsertResult, VectorRecord
from ..vector_store.service import VectorStoreService


@dataclass(slots=True)
class VectorIngestionService:
    """Coordinates ingestion workers writing into vector store namespaces."""

    vector_store: VectorStoreService
    namespace_map: dict[str, str] = field(default_factory=dict)
    batch_size: int = 256

    def register_dataset(self, dataset: str, namespace: str) -> None:
        self.namespace_map[dataset] = namespace

    def resolve_namespace(self, dataset: str) -> str:
        return self.namespace_map.get(dataset, dataset)

    def ingest_records(
        self,
        *,
        context: SecurityContext,
        dataset: str,
        records: Sequence[VectorRecord],
    ) -> UpsertResult:
        namespace = self.resolve_namespace(dataset)
        return self.vector_store.upsert(context=context, namespace=namespace, records=records)

    def ingest_documents(
        self,
        *,
        context: SecurityContext,
        dataset: str,
        documents: Sequence[Mapping[str, object]],
    ) -> UpsertResult:
        records = [self._to_record(document) for document in documents]
        return self.ingest_records(context=context, dataset=dataset, records=records)

    def ingest_batch(
        self,
        *,
        context: SecurityContext,
        dataset: str,
        records: Sequence[VectorRecord],
    ) -> UpsertResult:
        namespace = self.resolve_namespace(dataset)
        batches = []
        for start in range(0, len(records), max(self.batch_size, 1)):
            batches.append(records[start : start + self.batch_size])
        if len(batches) == 1:
            return self.vector_store.upsert(context=context, namespace=namespace, records=records)
        result = self.vector_store.bulk_upsert(
            context=context,
            namespace=namespace,
            batches=batches,
        )
        return result

    def _to_record(self, document: Mapping[str, object]) -> VectorRecord:
        vector_id = str(document.get("id"))
        values = list(document.get("vector", []))
        metadata = dict(document.get("metadata", {}))
        named_vectors = None
        if "named_vectors" in document:
            named_vectors = {
                name: list(values)
                for name, values in document["named_vectors"].items()  # type: ignore[index]
            }
        return VectorRecord(
            vector_id=vector_id,
            values=values,
            metadata=metadata,
            named_vectors=named_vectors,
        )


__all__ = ["VectorIngestionService"]
