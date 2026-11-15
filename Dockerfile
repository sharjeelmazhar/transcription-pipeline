# CPU image. For GPU, start from nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# and run with --gpus all.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stt/ stt/
COPY transcribe.py .

ENV WHISPER_MODEL=small STT_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "stt.api:app", "--host", "0.0.0.0", "--port", "8000"]
