FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    cryptography>=41.0.0 \
    grpcio>=1.60.0 \
    grpcio-tools>=1.60.0 \
    grpcio-health-checking>=1.60.0

# Set working directory
WORKDIR /app

# Copy certificate generation script
COPY scripts/generate_mtls_certificates.py /scripts/

# Create certificates directory
RUN mkdir -p /certs

# Set permissions
RUN chmod +x /scripts/generate_mtls_certificates.py

# Default command
CMD ["python", "/scripts/generate_mtls_certificates.py", "--help"]
