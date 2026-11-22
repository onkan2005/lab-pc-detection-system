# PC Usage Monitor: YOLO26 + face recognition web app (CPU).
# Personal data (photos, embeddings, logs) is NOT in the image; it is mounted at run time:
#   /enrollment  <- ./enrollment         (student photo folders, read-only)
#   /data        <- ./detector/data      (embeddings, assignments, logs, snapshots)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV runtime libraries; a compiler only while installing (insightface builds a small extension)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/detector
COPY detector/requirements.txt .
# CPU-only PyTorch first (much smaller than the default CUDA build), then the rest
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Run as a normal user (uid 1000 = the usual first user on Linux/WSL, so files written to the
# mounted data folder stay editable on the host)
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data /enrollment \
 && chown -R app:app /app /data /enrollment
USER app

# Download the pretrained models at build time so the container also works offline
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" \
 && python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])"

COPY --chown=app:app detector/ .

ENV PCMON_HOST=0.0.0.0 \
    PCMON_PORT=8765 \
    PCMON_DATA_DIR=/data \
    PCMON_ENROLLMENT_DIR=/enrollment

EXPOSE 8765
VOLUME ["/data"]
CMD ["python", "web.py"]
