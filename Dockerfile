FROM python:3.11-slim

# ── System deps: R runtime + packages needed to compile R extensions ──────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── R packages ────────────────────────────────────────────────────────────────
# Note: asreml is a commercial package (paid licence) — not available on CRAN
RUN R -e "install.packages(c('jsonlite','agricolae','dplyr','tidyr','readr','ggplot2','sommer','car','pbkrtest','lme4'), \
           repos='https://cloud.r-project.org/', dependencies=TRUE, Ncpus=2)"

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# ── Start ─────────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
