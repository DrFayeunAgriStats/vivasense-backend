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
    libgfortran5 \
    gfortran \
    build-essential \
    pkg-config \
    zlib1g-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy dependency manifests first for layer caching ────────────────────────
COPY requirements.txt ./

# ── Install Python dependencies ───────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Install R packages at build time (not at runtime) ─────────────────────────
RUN R -e "install.packages(c('Rcpp','cpp11','jsonlite'), repos='https://cloud.r-project.org/', dependencies=TRUE, Ncpus=2)" \
 && R -e "install.packages(c('lme4','pbkrtest','car'), repos='https://cloud.r-project.org/', dependencies=TRUE, Ncpus=2)" \
 && R -e "install.packages(c('agricolae','dplyr','tidyr','readr','ggplot2'), repos='https://cloud.r-project.org/', dependencies=TRUE, Ncpus=2)" \
 && R -e "install.packages('sommer', repos='https://cloud.r-project.org/', dependencies=TRUE, Ncpus=2)"

# Fail the image build immediately if 'car' is unavailable at runtime.
RUN Rscript -e "if (!requireNamespace('car', quietly=TRUE)) stop('R package car missing in image')"

# ── Copy the rest of the application ─────────────────────────────────────────
COPY . .

# ── Runtime configuration ─────────────────────────────────────────────────────
ENV PYTHONPATH=/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app_genetics:app --host 0.0.0.0 --port ${PORT:-8000}"]