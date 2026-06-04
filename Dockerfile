# Dockerfile for VSE (Video Subtitle Extractor) RunPod serverless endpoint.
#
# Build:
#   docker build -f Dockerfile.vse -t youruser/vse-runpod:latest .
#   docker push youruser/vse-runpod:latest
#
# RunPod setup:
#   Serverless → New Endpoint → paste your image → GPU: T4 or A10G
#   Min workers: 0, Max workers: 4 (matches VSE_PARALLEL_CHUNKS in .env)
#   Copy the endpoint ID → set RUNPOD_VSE_ENDPOINT in .env

FROM paddlepaddle/paddle:2.6.0-gpu-cuda11.7-cudnn8.4-trt8.4

# System deps: OpenCV headless, wget, git, ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# Clone VSE (shallow, saves ~150 MB)
RUN git clone --depth 1 \
    https://github.com/YaoFANGUK/video-subtitle-extractor.git \
    /app/vse

# Install VSE deps while preserving the Paddle 2.x GPU runtime in the base image.
# Upstream VSE now asks for PaddleOCR 3.x, but this worker code uses the 2.x API.
RUN grep -vE "^(paddlepaddle|paddleocr|pyside6|pyside6-fluent-widgets|je-showinfilemanager|numpy)" \
        /app/vse/requirements.txt > /tmp/requirements-vse-headless.txt && \
    pip install --no-cache-dir -r /tmp/requirements-vse-headless.txt && \
    pip install --no-cache-dir "numpy<2" "paddleocr==2.7.3" && \
    pip install --no-cache-dir runpod

# Copy RunPod handler
COPY handler.py /app/handler.py

ENV VSE_DIR=/app/vse

CMD ["python", "/app/handler.py"]
