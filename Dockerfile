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

# Clone VSE (shallow)
RUN git clone --depth 1 \
    https://github.com/YaoFANGUK/video-subtitle-extractor.git \
    /app/vse

# Install VSE deps + RunPod SDK
RUN pip install --no-cache-dir -r /app/vse/requirements.txt && \
    pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

ENV VSE_DIR=/app/vse

CMD ["python", "/app/handler.py"]
