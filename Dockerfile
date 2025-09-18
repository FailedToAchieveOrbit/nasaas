FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for production
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard] \
    prometheus-client

# Copy source code
COPY src/ ./src/
COPY setup.py .
COPY config/ ./config/
COPY examples/ ./examples/

# Install the package
RUN pip install -e .

# Create directories for data and cache
RUN mkdir -p /app/data /app/cache /app/logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV NASAAS_DATA_DIR=/app/data
ENV NASAAS_CACHE_DIR=/app/cache
ENV NASAAS_CONFIG_FILE=/app/config/default.yaml

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from nasaas import NASClient; client = NASClient(); print('OK')" || exit 1

# Default command
CMD ["nasaas", "--help"]

# Alternative: Run as web server
# CMD ["uvicorn", "nasaas.api:app", "--host", "0.0.0.0", "--port", "8000"]