# Torch-Dependent Code Archive

This directory contains archived torch-dependent code that was removed from the main codebase as part of the torch isolation architecture implementation.

## Archive Contents

### Original Torch-Dependent Files

- `gpu_manager_original.py` - Original GPU manager with torch dependencies
- `metrics_original.py` - Original metrics with torch-dependent GPU metrics
- `vector_store_gpu_original.py` - Original vector store GPU utilities
- `qwen3_service_original.py` - Original Qwen3 service with torch dependencies
- `splade_service_original.py` - Original SPLADE service with torch dependencies
- `reranking_batch_processor_original.py` - Original reranking batch processor
- `embeddings_gpu_utils_original.py` - Original embeddings GPU utilities

### Requirements Files

- `requirements_original.txt` - Original requirements.txt with torch dependencies
- `requirements_original.in` - Original requirements.in with torch dependencies

### Configuration Files

- `config_settings_original.py` - Original config settings with torch-related options
- `haystack_components_original.py` - Original Haystack components with torch dependencies

## Migration Notes

### What Was Replaced

1. **GPU Manager**: Replaced with `GpuServiceManager` that uses gRPC to communicate with GPU services
2. **Metrics**: Replaced with gRPC service metrics (`GPU_SERVICE_CALLS_TOTAL`, `GPU_SERVICE_CALL_DURATION_SECONDS`, etc.)
3. **Vector Store GPU**: Replaced with `VectorStoreGPU` that uses gRPC embedding services
4. **Qwen3 Service**: Replaced with gRPC embedding service calls
5. **SPLADE Service**: Replaced with gRPC embedding service calls
6. **Reranking**: Replaced with gRPC reranking service calls
7. **Embeddings GPU Utils**: Replaced with gRPC embedding service calls

### Architecture Changes

- **Before**: Direct torch usage in main gateway
- **After**: gRPC service calls to dedicated GPU containers

### Benefits of Migration

1. **Independent Scaling**: GPU services can scale independently
2. **Resource Isolation**: GPU memory and compute isolated to containers
3. **Simplified Deployment**: Main gateway doesn't require GPU dependencies
4. **Fail-Fast Behavior**: GPU service failures don't crash main gateway

## Restoration Instructions

If you need to restore any of this code for reference:

1. Copy the desired file from this archive
2. Remove the `_original` suffix
3. Place it in the appropriate location in the main codebase
4. Update imports and dependencies as needed
5. Test thoroughly to ensure compatibility

## Archive Date

Archived on: $(date)

## Archive Reason

Removed as part of torch isolation architecture implementation to move all GPU functionality to dedicated gRPC services.
