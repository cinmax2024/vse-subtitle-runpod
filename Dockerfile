FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 wget git libgomp1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip BEFORE installing Paddle — Ubuntu's bundled pip is too old to
# resolve Paddle's wheel metadata correctly and silently picks wrong builds.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PaddlePaddle GPU for CUDA 11.8.
# Use the cu118-specific index (-i), NOT the MKL/AVX page (-f) which is CPU-only.
# No version pin — gets the latest stable GPU wheel for cu118 (3.0.0 or newer).
RUN python -m pip install --no-cache-dir paddlepaddle-gpu \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

RUN git clone --depth 1 \
    https://github.com/YaoFANGUK/video-subtitle-extractor.git /app/vse

# Install paddleocr with --no-deps so pip cannot downgrade our GPU paddle
# to the CPU build when resolving paddleocr's declared paddle dependency.
RUN python -m pip install --no-cache-dir "paddleocr~=3.4.0" --no-deps

# Remaining VSE deps — three package groups are skipped:
#   paddlepaddle / paddleocr — already installed above
#   pyside6 / pyside6-fluent-widgets — Qt6 GUI, requires a display, fails headless
#   je-showinfilemanager — desktop file-manager helper, also fails headless
RUN grep -vE "^(paddlepaddle|paddleocr|pyside6|je-showinfilemanager)" \
        /app/vse/requirements.txt > /tmp/req.txt && \
    python -m pip install --no-cache-dir -r /tmp/req.txt

RUN python -m pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

ENV VSE_DIR=/app/vse

CMD ["python", "/app/handler.py"]
