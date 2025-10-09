"""Constant docstring template for pipeline constants.

This template shows the required structure and content for module-level
constant docstrings in the Medical_KG_rev pipeline codebase.
"""

# Example constant docstring structure:

# [One-line summary of what the constant represents].
#
# [Detailed explanation of what the constant is used for, when to use it,
# and any constraints or valid values.]
#
# Usage:
#     [Show how to use the constant]
#     [Include examples of common patterns]
#
# Note:
#     [Any important implementation notes]
#     [Performance considerations]
#     [Thread safety information]
#
# Example:
#     >>> if value == MY_CONSTANT:
#     ...     process_special_case()
#     >>> print(f"Using constant: {MY_CONSTANT}")
#     Using constant: constant_value

# Real example for chunking constants:

# Default chunking strategy for documents without explicit strategy.
#
# This constant defines the default chunking strategy used when no
# strategy is specified in a chunking request. It provides a sensible
# default that works well for most document types.
#
# Usage:
#     Use this constant when setting default strategy values or
#     when validating strategy parameters in chunking requests.
#
# Note:
#     This value must match one of the strategies available in
#     the ChunkingService.available_strategies property.
#
# Example:
#     >>> strategy = request.strategy or DEFAULT_CHUNKING_STRATEGY
#     >>> if strategy not in available_strategies:
#     ...     strategy = DEFAULT_CHUNKING_STRATEGY
DEFAULT_CHUNKING_STRATEGY = "section"

# Maximum number of tokens per chunk for safety and performance.
#
# This constant defines the maximum number of tokens that can be
# included in a single chunk to prevent memory issues and ensure
# reasonable processing times. Chunks exceeding this limit will
# be split further.
#
# Usage:
#     Use this constant when validating chunk size parameters
#     or when setting upper bounds for chunking operations.
#
# Note:
#     This value is enforced by the ChunkingService and cannot
#     be exceeded even if requested in chunking parameters.
#
# Example:
#     >>> chunk_size = min(request.chunk_size, MAX_CHUNK_SIZE)
#     >>> if chunk_size > MAX_CHUNK_SIZE:
#     ...     raise ValueError(f"Chunk size exceeds maximum: {MAX_CHUNK_SIZE}")
MAX_CHUNK_SIZE = 2048

# Default token overlap between adjacent chunks.
#
# This constant defines the default number of tokens that overlap
# between adjacent chunks to ensure context preservation and
# improve retrieval accuracy.
#
# Usage:
#     Use this constant when setting default overlap values or
#     when validating overlap parameters in chunking requests.
#
# Note:
#     This value should be reasonable (typically 10-20% of chunk size)
#     to balance context preservation with storage efficiency.
#
# Example:
#     >>> overlap = request.overlap or DEFAULT_CHUNK_OVERLAP
#     >>> if overlap < 0 or overlap > chunk_size:
#     ...     overlap = DEFAULT_CHUNK_OVERLAP
DEFAULT_CHUNK_OVERLAP = 128

# Real example for embedding constants:

# Default embedding model for documents without explicit model.
#
# This constant defines the default embedding model used when no
# model is specified in an embedding request. It provides a
# sensible default that works well for most text types.
#
# Usage:
#     Use this constant when setting default model values or
#     when validating model parameters in embedding requests.
#
# Note:
#     This value must match one of the models available in
#     the EmbeddingModelRegistry.available_models property.
#
# Example:
#     >>> model_name = request.model_name or DEFAULT_EMBEDDING_MODEL
#     >>> if model_name not in available_models:
#     ...     model_name = DEFAULT_EMBEDDING_MODEL
DEFAULT_EMBEDDING_MODEL = "biobert-base-cased-v1.1"

# Maximum number of texts that can be embedded in a single batch.
#
# This constant defines the maximum number of texts that can be
# processed in a single embedding batch to prevent memory issues
# and ensure reasonable processing times.
#
# Usage:
#     Use this constant when validating batch size parameters
#     or when setting upper bounds for embedding operations.
#
# Note:
#     This value is enforced by the EmbeddingService and cannot
#     be exceeded even if requested in embedding parameters.
#
# Example:
#     >>> batch_size = min(len(texts), MAX_EMBEDDING_BATCH_SIZE)
#     >>> if len(texts) > MAX_EMBEDDING_BATCH_SIZE:
#     ...     # Split into multiple batches
#     ...     batches = [texts[i:i+MAX_EMBEDDING_BATCH_SIZE]
#     ...              for i in range(0, len(texts), MAX_EMBEDDING_BATCH_SIZE)]
MAX_EMBEDDING_BATCH_SIZE = 100

# Default embedding dimension for models without explicit dimension.
#
# This constant defines the default embedding dimension used when
# the model dimension is not explicitly specified or cannot be
# determined from the model configuration.
#
# Usage:
#     Use this constant when setting default dimension values or
#     when validating dimension parameters in embedding requests.
#
# Note:
#     This value should match the actual dimension of the default
#     embedding model to avoid dimension mismatches.
#
# Example:
#     >>> dimension = request.dimension or DEFAULT_EMBEDDING_DIMENSION
#     >>> if dimension != expected_dimension:
#     ...     dimension = DEFAULT_EMBEDDING_DIMENSION
DEFAULT_EMBEDDING_DIMENSION = 768

# Real example for orchestration constants:

# Default timeout for orchestration stage execution.
#
# This constant defines the default timeout in seconds for
# orchestration stage execution. Stages that exceed this
# timeout will be marked as failed.
#
# Usage:
#     Use this constant when setting default timeout values or
#     when validating timeout parameters in stage execution.
#
# Note:
#     This value should be reasonable for most stages while
#     preventing runaway processes from consuming resources.
#
# Example:
#     >>> timeout = stage_config.timeout or DEFAULT_STAGE_TIMEOUT
#     >>> if timeout > MAX_STAGE_TIMEOUT:
#     ...     timeout = DEFAULT_STAGE_TIMEOUT
DEFAULT_STAGE_TIMEOUT = 300  # 5 minutes

# Maximum number of retries for failed orchestration stages.
#
# This constant defines the maximum number of retries allowed
# for failed orchestration stages before marking them as
# permanently failed.
#
# Usage:
#     Use this constant when setting default retry values or
#     when validating retry parameters in stage execution.
#
# Note:
#     This value should balance reliability with resource
#     consumption and processing time.
#
# Example:
#     >>> max_retries = stage_config.max_retries or DEFAULT_MAX_RETRIES
#     >>> if max_retries > MAX_ALLOWED_RETRIES:
#     ...     max_retries = DEFAULT_MAX_RETRIES
DEFAULT_MAX_RETRIES = 3

# Default batch size for processing multiple items in orchestration stages.
#
# This constant defines the default batch size for processing
# multiple items in orchestration stages to balance efficiency
# with memory usage.
#
# Usage:
#     Use this constant when setting default batch size values or
#     when validating batch size parameters in stage execution.
#
# Note:
#     This value should be reasonable for most stages while
#     preventing memory issues with large batches.
#
# Example:
#     >>> batch_size = stage_config.batch_size or DEFAULT_BATCH_SIZE
#     >>> if batch_size > MAX_BATCH_SIZE:
#     ...     batch_size = DEFAULT_BATCH_SIZE
DEFAULT_BATCH_SIZE = 50
