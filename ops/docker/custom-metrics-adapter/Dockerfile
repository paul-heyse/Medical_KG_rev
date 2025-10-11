FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY custom_metrics_adapter.py .

# Create non-root user
RUN useradd -m -u 1000 adapter && chown -R adapter:adapter /app
USER adapter

# Expose port
EXPOSE 8080

# Start custom metrics adapter
CMD ["python", "custom_metrics_adapter.py", "--host", "0.0.0.0", "--port", "8080"]
