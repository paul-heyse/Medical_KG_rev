# Simulation Artifact Catalog

This document catalogs all simulation artifacts that need to be removed as part of Phase 4 of the observability architecture modernization.

## Files to Delete

### src/Medical_KG_rev/services/mineru/vllm_client.py

- **Lines**: 1-61
- **Type**: Mock VLLMClient simulation
- **Used By**: No external imports found
- **Replacement**: Real gRPC client (already implemented in Phase 3)
- **Issues**:
  - Simulates VLLM generation with `asyncio.sleep(0.5)`
  - Returns mock responses instead of real embeddings
  - Masks actual VLLM connectivity issues

### src/Medical_KG_rev/services/mineru/cli_wrapper.py

- **Lines**: 1-121
- **Type**: Simulation wrapper for MinerU CLI
- **Used By**: No external imports found
- **Replacement**: Real MinerU CLI integration or proper mocks at gRPC level
- **Issues**:
  - `SimulatedMineruCli` class simulates document processing
  - `MineruCliWrapper` simulates CLI processing with `asyncio.sleep(2.0)`
  - Masks actual MinerU CLI availability and performance

### src/Medical_KG_rev/services/clients/test_client.py

- **Lines**: 1-576
- **Type**: Mock service client for testing
- **Used By**: No external imports found
- **Replacement**: Proper test mocks using unittest.mock or testcontainers
- **Issues**:
  - `MockServiceClient` simulates gRPC service calls
  - `ServiceClientTester` provides simulation-based testing utilities
  - Masks actual service integration gaps

## Integration Gaps Masked

### 1. VLLM Connection Issues

- **Current**: Simulation always returns "healthy" status
- **Reality**: VLLM service may be unavailable, have GPU issues, or network problems
- **Impact**: Tests pass but production fails silently

### 2. MinerU Processing Performance

- **Current**: Simulation uses fixed delays (0.1s, 2.0s)
- **Reality**: Processing time varies based on document complexity, GPU availability, model loading
- **Impact**: Performance tests don't reflect real-world behavior

### 3. GPU Service Availability

- **Current**: Simulation always returns "gpu_available": True
- **Reality**: GPU services may be unavailable, overloaded, or have memory issues
- **Impact**: Circuit breaker logic never tested with real failures

### 4. Service Discovery and Load Balancing

- **Current**: Mock clients don't test actual service discovery
- **Reality**: Services may be unavailable, have different endpoints, or require load balancing
- **Impact**: Production deployment issues not caught in testing

## Replacement Strategy

### For VLLM Integration

- **Use**: Real gRPC client implemented in Phase 3 (`Qwen3GRPCClient`)
- **Testing**: Use testcontainers to start real vLLM service for integration tests
- **Mocking**: Mock at gRPC level using `unittest.mock` for unit tests

### For MinerU Integration

- **Use**: Real MinerU CLI wrapper or gRPC service
- **Testing**: Use testcontainers for integration tests
- **Mocking**: Mock CLI subprocess calls for unit tests

### For Service Client Testing

- **Use**: `unittest.mock` for unit tests
- **Use**: testcontainers for integration tests
- **Use**: Real service clients for end-to-end tests

## Files That Reference Simulation Artifacts

### No External Dependencies Found

- None of the simulation files are imported by production code
- All simulation artifacts are self-contained
- Safe to delete without breaking existing functionality

## Testing Impact

### Current Test Coverage

- Simulation artifacts provide false test coverage
- Tests pass but don't validate real service integration
- Performance tests use unrealistic timing

### Post-Removal Testing Strategy

1. **Unit Tests**: Use `unittest.mock` to mock external dependencies
2. **Integration Tests**: Use testcontainers to start real services
3. **Performance Tests**: Use real services or realistic mocks
4. **End-to-End Tests**: Use real service clients

## Migration Steps

1. **Delete simulation files** (Tasks 4.2.1)
2. **Update integration tests** to use testcontainers (Task 4.3.1)
3. **Verify no broken imports** in production code
4. **Update CI/CD** to use real services for integration tests
5. **Document testing strategy** for future development

## Benefits of Removal

1. **Real Integration Testing**: Tests will catch actual service integration issues
2. **Accurate Performance Metrics**: Performance tests will reflect real-world behavior
3. **Better Error Handling**: Circuit breakers and error handling will be properly tested
4. **Reduced Technical Debt**: Elimination of simulation code reduces maintenance burden
5. **Improved Reliability**: Production deployments will be more reliable due to better testing
