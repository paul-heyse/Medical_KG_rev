# Torch-Free Main Gateway Docker Image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies (excluding torch)
RUN pip install --no-cache-dir -r requirements.txt

# Install gRPC dependencies
RUN pip install \
    grpcio \
    grpcio-tools \
    grpcio-health-checking

# Copy application code
COPY src/Medical_KG_rev/ /app/Medical_KG_rev/
COPY src/prometheus_client/ /app/prometheus_client/
COPY src/structlog/ /app/structlog/

# Copy configuration
COPY config/ /app/config/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set default command
CMD ["python", "-m", "Medical_KG_rev.gateway.main"]
