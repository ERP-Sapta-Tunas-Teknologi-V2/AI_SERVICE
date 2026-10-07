# =======================================================
# Production Dockerfile for AI_SERVICE (MoM & AI Service)
# =======================================================
FROM python:3.11-slim

# Set environment variables for Python, Timezone & HuggingFace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Jakarta \
    PORT=5000 \
    GUNICORN_WORKERS=2 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=300 \
    HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies required by audio processing & ML libraries (FFmpeg, libsndfile, PyAV, Torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-privileged user and group for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    pip install --no-cache-dir torch torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure upload, cache, and log directories exist with correct permissions
RUN mkdir -p /app/temp_audio /app/log /app/.cache/huggingface && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose internal service port
EXPOSE 5000

# Container healthcheck
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

# Graceful stop signal
STOPSIGNAL SIGTERM

# Run application using Gunicorn with gthread worker class (supports SSE streaming)
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --worker-class gthread --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-300} --graceful-timeout 30 --keep-alive 5 --access-logfile - --error-logfile - app:app"]
