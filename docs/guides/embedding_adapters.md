# Embedding Adapter Developer Guide

This guide describes the process for adding new embedders to the universal embedding subsystem. All adapters must implement the
`BaseEmbedder` protocol and return `EmbeddingRecord` instances that conform to namespace governance rules.

## 1. Choose the Namespace

Namespaces follow the pattern `{kind}.{model}.{dim}.{version}`. Dense single-vector adapters typically use a `single_vector`
prefix while multi-vector adapters use `multi_vector`. If your adapter can auto-discover dimensionality, set the namespace dim
segment to `auto` and rely on the namespace manager to record the observed value.

## 2. Implement the Adapter

1. Import the protocol and registry helpers:
   ```python
   from Medical_KG_rev.embeddings.ports import BaseEmbedder, EmbedderConfig, EmbeddingRecord, EmbeddingRequest
   from Medical_KG_rev.embeddings.registry import EmbedderRegistry
   ```
2. Implement the adapter class with `embed_documents()` and `embed_queries()` methods.
3. Populate `EmbeddingRecord.metadata` with provider information and any adapter-specific fields (e.g. offsets, shard IDs).
4. Register the adapter in `Medical_KG_rev.embeddings.providers.register_builtin_embedders()`.

## 3. Dimension Validation

The `NamespaceManager` validates dimensionality for every record. Ensure the adapter sets the `dim` field for dense vectors or
stores the effective dimension in the first vector. Sparse adapters should populate `terms` and neural sparse adapters should set
`neural_fields`.

## 4. Batch Processing and Progress Reporting

Adapters that support batch inference should use the batching utilities in `Medical_KG_rev.embeddings.utils.batching`. These
helpers provide progress callbacks that log batch completion without requiring external libraries.

## 5. Testing Checklist

- Unit test the adapter in isolation with deterministic embeddings.
- Add contract tests verifying compliance with the `BaseEmbedder` protocol.
- If the adapter introduces new storage targets, extend `StorageRouter` with custom handlers.

For complete examples, review the dense, sparse, multi-vector, and neural sparse adapters in the `embeddings/` package.
