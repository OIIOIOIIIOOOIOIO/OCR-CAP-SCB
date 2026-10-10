# Captcha OCR inference server — CPU image (small, portable).
FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached layer). torch CPU wheels come from the pytorch index.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App + model weights
COPY serve.py .
COPY models/ ./models/

ENV HOST=0.0.0.0 \
    PORT=8765 \
    TTA=0 \
    MODEL_DIR=/app/models \
    OMP_NUM_THREADS=4

EXPOSE 8765

# liveness: GET / should return 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/',timeout=4).status==200 else 1)"

CMD ["python", "serve.py"]
