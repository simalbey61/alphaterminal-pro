# ============================================================================
# AlphaTerminal Pro - Backend Dockerfile
# ============================================================================
# Multi-stage build for optimized production image
# ============================================================================

# =============================================================================
# STAGE 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# =============================================================================
# STAGE 2: Production
# =============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="AlphaTerminal Team <support@alphaterminal.com>" \
      version="1.0.0" \
      description="AlphaTerminal Pro Backend API"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    # Application
    APP_NAME="AlphaTerminal Pro" \
    ENVIRONMENT=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 alphaterminal && \
    useradd --uid 1000 --gid alphaterminal --shell /bin/bash --create-home alphaterminal

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=alphaterminal:alphaterminal . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/data/cache /app/data/models && \
    chown -R alphaterminal:alphaterminal /app

# Switch to non-root user
USER alphaterminal

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]


# =============================================================================
# STAGE 3: Development
# =============================================================================
FROM production as development

# Switch to root for installing dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install dev dependencies
RUN pip install \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-asyncio \
    pytest-cov

# Switch back to non-root user
USER alphaterminal

# Development command with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
