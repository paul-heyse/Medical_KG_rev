# Chunking Adapter Developer Guide

This guide describes how to integrate new third-party chunking adapters with the modular chunking
system. Adapters wrap external frameworks and expose them through the `BaseChunker` protocol so that
chunkers can be configured declaratively.

## Design Principles

* **Provenance first** – adapters must populate `Chunk.meta['block_ids']` so downstream components can
  trace chunks back to their origin blocks.
* **Offset fidelity** – adapters must translate framework offsets to character offsets relative to the
  `Document` input using the shared `OffsetMapper` helper.
* **Graceful degradation** – adapters should raise `ChunkerConfigurationError` when optional
  dependencies are missing and avoid importing heavy frameworks at module import time.

## Implementation Steps

1. **Create adapter module** – add a class that implements `BaseChunker` and wraps the target framework.
2. **Resolve framework splitter/parser** – lazily import the framework inside the constructor to keep
   import failures isolated.
3. **Map outputs to contexts** – use `OffsetMapper` to project framework chunks back to `BlockContext`
   instances before calling `ChunkAssembler`.
4. **Register the adapter** – add the adapter to `chunking/registry.py` so it can be referenced from
   configuration files.
5. **Document parameters** – update `docs/chunking/Chunkers.md` with configuration hints and defaults.

## Testing Checklist

* Add unit tests under `tests/chunking/test_framework_adapters.py` using
  `pytest.importorskip` to execute the adapter when the dependency is available.
* Include at least one end-to-end evaluation through the `ChunkingEvaluationRunner` to verify chunk
  quality and boundary behaviour.
