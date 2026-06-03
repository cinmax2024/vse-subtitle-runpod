FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 wget git libgomp1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && rm -rf /var/lib/apt/lists/*

# Step 1 — PaddlePaddle GPU for CUDA 11.8 (installed BEFORE paddleocr so pip
# cannot downgrade it to the CPU build when resolving paddleocr's dependencies)
RUN pip install --no-cache-dir paddlepaddle-gpu==3.0.0.post118 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

RUN git clone --depth 1 \
    https://github.com/YaoFANGUK/video-subtitle-extractor.git /app/vse

# Step 2 — paddleocr with --no-deps so pip does NOT pull the CPU paddle on top
# (paddlepaddle-gpu 3.0.0 is API-compatible with the ~=3.3 requirement)
RUN pip install --no-cache-dir "paddleocr~=3.4.0" --no-deps

# Step 3 — remaining VSE deps, skipping:
#   paddlepaddle        — GPU version already installed above
#   paddleocr           — already installed above
#   pyside6 / pyside6-fluent-widgets — Qt6 GUI, requires a display; headless RunPod fails
#   je-showinfilemanager — desktop file-manager helper, not installable headless
RUN grep -vE "^(paddlepaddle|paddleocr|pyside6|je-showinfilemanager)" \
        /app/vse/requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt

RUN pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

ENV VSE_DIR=/app/vse

CMD ["python", "/app/handler.py"]
