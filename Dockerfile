FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential tesseract-ocr tesseract-ocr-khm \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the dictionary at image build time so cold starts do not download it.
RUN python scripts/build_dict.py

EXPOSE 7860
# Cloud Run injects $PORT and health-checks it; Spaces expects 7860.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
