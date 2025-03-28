FROM python:3.9-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip packages with exact versions
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.68.1 \
    uvicorn==0.15.0 \
    onnxruntime==1.10.0 \
    huggingface-hub==0.4.0 \
    numpy==1.21.2 \
    pydantic==1.8.2

# Copy application code
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]