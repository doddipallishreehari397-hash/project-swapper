# Use an official lightweight Python base image optimized for deep learning runtimes
FROM python:3.10-slim

# Prevent Python from writing pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install essential system binary dependencies and process managers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    supervisor \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency configuration and install pipelines
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories for app cache and model weight residency
RUN mkdir -p temp_cache models

# Copy source architecture codebases into container workspace
COPY app.py cache_cleaner.py entrypoint.sh .
COPY models/inswapper_128.onnx models/
RUN chmod +x entrypoint.sh

# Configure process monitor supervisor configuration
RUN echo '[supervisord]\nnodaemon=true\n\n[program:streamlit]\ncommand=streamlit run app.py --server.port=8501 --server.address=0.0.0.0\nautostart=true\nautorestart=true\n\n[program:cache_cleaner]\ncommand=python cache_cleaner.py\nautostart=true\nautorestart=true\n' > /etc/supervisor/conf.d/supervisord.conf

# Expose standard default Streamlit structural application layer port
EXPOSE 8501

# Boot application processes through process engine monitor
ENTRYPOINT ["/app/entrypoint.sh"]
