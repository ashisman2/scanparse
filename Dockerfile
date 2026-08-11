# ScanParse: lightweight, security-hardened container
# Non-root user, pinned OS packages, slim base image.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Tesseract + Hindi/English language packs + poppler (PDF->image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-hin \
        tesseract-ocr-eng \
        poppler-utils \
        libgl1 \
        libgomp1 \
        curl && \
    rm -rf /var/lib/apt/lists/*
# Overlay the tessdata_best Hindi model (LSTM) for far better Devanagari accuracy
RUN cd /usr/share/tesseract-ocr/*/tessdata && \
    curl -sSL -o hin.traineddata \
        https://github.com/tesseract-ocr/tessdata_best/raw/main/hin.traineddata

RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[ui]" && \
    pip cache purge

COPY scanparse/ scanparse/

# Drop privileges: run as non-root user
USER appuser

EXPOSE 7860
CMD ["python", "-m", "scanparse.app"]
