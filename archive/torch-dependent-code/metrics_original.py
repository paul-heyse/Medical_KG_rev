"""Prometheus metrics for GPU services."""

from prometheus_client import Counter, Gauge, Histogram

# GPU metrics (torch-dependent)
GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_mb", "GPU memory usage in MB", ["device_id", "device_name"]
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_percentage", "GPU utilization percentage", ["device_id", "device_name"]
)

GPU_TEMPERATURE = Gauge(
    "gpu_temperature_celsius", "GPU temperature in Celsius", ["device_id", "device_name"]
)

GPU_POWER_USAGE = Gauge(
    "gpu_power_usage_watts", "GPU power usage in watts", ["device_id", "device_name"]
)

# Model loading metrics
MODEL_LOAD_TIME = Histogram(
    "model_load_time_seconds", "Time taken to load models", ["model_name", "model_type"]
)

MODEL_INFERENCE_TIME = Histogram(
    "model_inference_time_seconds", "Time taken for model inference", ["model_name", "model_type"]
)

# Batch processing metrics
BATCH_PROCESSING_TIME = Histogram(
    "batch_processing_time_seconds", "Time taken for batch processing", ["batch_size", "model_name"]
)

BATCH_PROCESSING_THROUGHPUT = Counter(
    "batch_processing_throughput_total",
    "Total number of items processed in batches",
    ["model_name"],
)

# Error metrics
GPU_ERRORS = Counter("gpu_errors_total", "Total number of GPU errors", ["error_type", "device_id"])

MODEL_ERRORS = Counter(
    "model_errors_total", "Total number of model errors", ["model_name", "error_type"]
)
