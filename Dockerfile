FROM python:3.11-slim

# ── System deps: R runtime + compilation tools ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    cmake \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    liblapack-dev \
    libopenblas-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libcairo2-dev \
    libglpk-dev \
    libxt-dev \
    libx11-dev \
    gfortran \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy dependency manifests first for layer caching ────────────────────────
COPY requirements.txt install_r_packages.R ./

# ── Install Python dependencies ───────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Install R packages at build time (not at runtime) ─────────────────────────
RUN Rscript install_r_packages.R

# ── Copy the rest of the application ─────────────────────────────────────────
COPY . .

# ── Runtime configuration ─────────────────────────────────────────────────────
ENV PYTHONPATH=/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]