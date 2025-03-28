FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy aplikasi
COPY . .

# Gunakan salah satu format CMD di bawah ini:
CMD uvicorn main:app --host 0.0.0.0 --port 8000  # Pilihan 1 (Shell Form)
# atau:
# ENTRYPOINT ["uvicorn"]                         # Pilihan 2 (Exec Form)
# CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]