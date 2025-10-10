# Coordinators API

The coordinators layer provides a protocol-agnostic interface between the gateway services and domain logic. Each coordinator manages a specific type of operation (chunking, embedding) and handles job lifecycle, error translation, and metrics emission.

## Chunking Coordinator

The `ChunkingCoordinator` coordinates synchronous chunking operations by managing job lifecycle, delegating to ChunkingService, and translating errors.

::: Medical_KG_rev.gateway.coordinators.chunking.ChunkingCoordinator

## Embedding Coordinator

The `EmbeddingCoordinator` coordinates embedding operations with policy enforcement, namespace resolution, and persistence.

::: Medical_KG_rev.gateway.coordinators.embedding.EmbeddingCoordinator

## Base Coordinator

The base coordinator abstractions define the common interface and configuration for all coordinators.

::: Medical_KG_rev.gateway.coordinators.base.CoordinatorConfig
::: Medical_KG_rev.gateway.coordinators.base.CoordinatorRequest
::: Medical_KG_rev.gateway.coordinators.base.CoordinatorResult
::: Medical_KG_rev.gateway.coordinators.base.BaseCoordinator

## Job Lifecycle Manager

The `JobLifecycleManager` tracks job states and publishes SSE events for real-time status updates.

::: Medical_KG_rev.gateway.coordinators.job_lifecycle.JobLifecycleManager
