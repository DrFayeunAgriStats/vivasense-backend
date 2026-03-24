FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages
RUN R -e "install.packages(c('jsonlite','dplyr','readr'), repos='https://cloud.r-project.org/')"

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Start server
CMD ["python", "-m", "uvicorn", "app_genetics:app", "--host", "0.0.0.0", "--port", "8000"]
