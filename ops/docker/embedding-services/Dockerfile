# Embedding Services Docker Container with Torch Ecosystem
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install embedding libraries
RUN pip install \
    transformers \
    accelerate \
    sentence-transformers \
    numpy \
    scipy

# Install gRPC dependencies
RUN pip install \
    grpcio \
    grpcio-tools \
    grpcio-health-checking

# Install application dependencies
RUN pip install \
    structlog \
    prometheus-client \
    pydantic \
    httpx

# Create application directory
WORKDIR /app

# Copy embedding service code
COPY src/Medical_KG_rev/services/embedding/ /app/services/embedding/
COPY src/Medical_KG_rev/proto/ /app/proto/
COPY src/Medical_KG_rev/observability/ /app/observability/
COPY src/Medical_KG_rev/models/ /app/models/
COPY src/Medical_KG_rev/config/ /app/config/

# Create model cache directory
RUN mkdir -p /app/models/cache

# Compile protobuf files
RUN python -m grpc_tools.protoc \
    --proto_path=/app/proto \
    --python_out=/app \
    --grpc_python_out=/app \
    /app/proto/embedding.proto

# Create gRPC server entry point
COPY ops/docker/embedding-services/grpc_server.py /app/grpc_server.py

# Health check using gRPC health protocol
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; from grpc_health.v1 import health_pb2_grpc, health_pb2; \
    channel = grpc.insecure_channel('localhost:50051'); \
    stub = health_pb2_grpc.HealthStub(channel); \
    response = stub.Check(health_pb2.HealthCheckRequest(service='')); \
    assert response.status == health_pb2.HealthCheckResponse.SERVING"

# Expose gRPC port
EXPOSE 50051

# Set default command
CMD ["python", "/app/grpc_server.py"]
