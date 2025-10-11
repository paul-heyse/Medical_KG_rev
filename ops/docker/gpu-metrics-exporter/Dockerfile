FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nvidia-ml-py3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY gpu_metrics_exporter.py .

# Create non-root user
RUN useradd -m -u 1000 metrics && chown -R metrics:metrics /app
USER metrics

# Expose metrics port
EXPOSE 8000

# Start metrics exporter
CMD ["python", "gpu_metrics_exporter.py", "--service", "gpu-service", "--port", "8000"]
