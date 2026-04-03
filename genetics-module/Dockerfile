FROM python:3.11-slim

# Install system dependencies (including gfortran for compiled R packages like sommer)
RUN apt-get update && apt-get install -y \
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

# Install required R packages
# Note: asreml is a commercial package requiring a paid license — not installable from CRAN
RUN R -e "install.packages(c('jsonlite','agricolae','dplyr','tidyr','readr','ggplot2','sommer'), repos='https://cloud.r-project.org/', Ncpus=2)"

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
# python-docx is listed in requirements.txt; this explicit install ensures it
# is present even if Docker layer cache serves a stale requirements install.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "python-docx==1.1.2"

# Expose port
EXPOSE 8000

# Start server
CMD ["sh", "-c", "uvicorn app_genetics:app --host 0.0.0.0 --port ${PORT:-8000}"]
