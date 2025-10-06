# gRPC Services Specification

## ADDED Requirements

### Requirement: Protocol Buffer Definitions

The system SHALL define all gRPC services using Protocol Buffers (.proto files) with Buf validation.

#### Scenario: MinerU service definition

- **WHEN** mineru.proto is defined
- **THEN** it MUST include ProcessPDF RPC with request/response messages

#### Scenario: Buf lint passing

- **WHEN** running `buf lint`
- **THEN** all .proto files MUST pass without errors

### Requirement: gRPC Service Implementation

The system SHALL implement gRPC servers for GPU-bound and internal operations.

#### Scenario: Embedding service

- **WHEN** gRPC client calls EmbedChunks
- **THEN** service MUST generate embeddings and return vectors

#### Scenario: Health check

- **WHEN** calling health check RPC
- **THEN** service MUST return status and version

### Requirement: Code Generation

The system SHALL generate Python code from .proto files using Buf.

#### Scenario: Automatic code gen

- **WHEN** .proto file is updated
- **THEN** Python stubs MUST be regenerated in CI

### Requirement: Breaking Change Detection

The system SHALL prevent accidental breaking changes to gRPC services using Buf.

#### Scenario: Field removal detection

- **WHEN** removing a field from proto
- **THEN** `buf breaking` MUST fail and prevent merge
