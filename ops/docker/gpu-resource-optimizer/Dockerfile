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
COPY gpu_resource_optimizer.py .
COPY optimize_gpu_resources.py .

# Create non-root user
RUN useradd -m -u 1000 optimizer && chown -R optimizer:optimizer /app
USER optimizer

# Expose metrics port
EXPOSE 8000

# Start GPU resource optimizer
CMD ["python", "gpu_resource_optimizer.py", "--interval", "300"]
