FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libgl1-mesa-glx libglib2.0-0 wget git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 \
    https://github.com/YaoFANGUK/video-subtitle-extractor.git /app/vse

# Three packages in requirements.txt break headless Docker builds:
#   paddlepaddle~=3.3       — 3.3 not on PyPI (latest is 3.0.0); conflicts with
#                             the 2.6.0 pre-installed in the paddle base image.
#                             Removed here so paddleocr pulls the right version
#                             as its own transitive dependency.
#   pyside6 / pyside6-fluent-widgets — Qt6 GUI libs that require a display.
#                             RunPod containers are headless; these always fail.
#   je-showinfilemanager    — desktop file-manager helper, not installable headless.
RUN grep -vE "^(paddlepaddle|pyside6|je-showinfilemanager)" /app/vse/requirements.txt \
    > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt && \
    pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

ENV VSE_DIR=/app/vse

CMD ["python", "/app/handler.py"]
