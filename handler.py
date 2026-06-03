"""
RunPod serverless handler — burned-in subtitle OCR via PaddleOCR.

Bypasses VSE's interactive main.py entirely. Instead:
  1. Pre-crops video to bottom 22% (subtitle zone) — eliminates logos/watermarks.
  2. Extracts frames at 2 fps with ffmpeg.
  3. Runs PaddleOCR on each frame (GPU-accelerated).
  4. Groups consecutive frames with identical text into SRT cues.

Input:  {"video_url": "<R2 presigned URL>", "language": "en"}
Output: {"ok": true, "srt": "<srt text>", "cue_count": N}
        {"ok": false, "error": "<message>"}
"""

import glob
import logging
import os
import subprocess
import tempfile

import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("handler")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tc(ms):
    ms = max(0, int(ms))
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _download(url, dest):
    r = subprocess.run(
        ["wget", "-q", "--timeout=120", "-O", dest, url],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"wget: {r.stderr[:400]}")


def _crop_subtitle_zone(src, dest, ratio=0.22):
    """Keep only the bottom `ratio` of the frame — subtitle lives here."""
    top = 1.0 - ratio
    r = subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-vf", f"crop=in_w:in_h*{ratio}:0:in_h*{top}",
        "-c:a", "copy", dest,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"crop: {r.stderr[:300]}")


def _extract_frames(video, frames_dir, fps=2):
    """Dump frames at `fps` into frames_dir as JPEGs."""
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video,
        "-vf", f"fps={fps}",
        os.path.join(frames_dir, "f%06d.jpg"),
    ], check=True)


def _ocr_frames(frames_dir, language, fps=2):
    """
    Run PaddleOCR on every frame, return list of (timestamp_ms, text).
    GPU is used automatically when available.
    """
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang=language,
        show_log=False,
        use_gpu=True,
    )

    ms_per_frame = int(1000 / fps)
    rows = []

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    logger.info("OCR on %d frames...", len(frames))

    for idx, path in enumerate(frames):
        ts = idx * ms_per_frame
        try:
            result = ocr.ocr(path, cls=False)
        except Exception as exc:
            logger.warning("OCR frame %d error: %s", idx, exc)
            continue
        if not result or not result[0]:
            continue
        parts = []
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            text, conf = line[1][0], line[1][1]
            if conf >= 0.6 and len(text.strip()) >= 2:
                parts.append(text.strip())
        if parts:
            rows.append((ts, " ".join(parts)))

    return rows


def _rows_to_srt(rows, fps=2):
    """Merge consecutive identical text into SRT cues."""
    if not rows:
        return ""

    ms_per_frame = int(1000 / fps)
    cues = []
    cur_text  = rows[0][1]
    cur_start = rows[0][0]
    cur_end   = rows[0][0] + ms_per_frame

    for ts, text in rows[1:]:
        if text == cur_text and ts <= cur_end + ms_per_frame * 2:
            cur_end = ts + ms_per_frame
        else:
            if cur_end - cur_start >= 300:
                cues.append((cur_start, cur_end, cur_text))
            cur_text  = text
            cur_start = ts
            cur_end   = ts + ms_per_frame

    if cur_end - cur_start >= 300:
        cues.append((cur_start, cur_end, cur_text))

    lines = []
    for i, (s, e, t) in enumerate(cues, 1):
        lines += [str(i), f"{_tc(s)} --> {_tc(e)}", t, ""]
    return "\n".join(lines)


# ── Handler ───────────────────────────────────────────────────────────────────

def handler(job):
    inp       = job.get("input", {})
    video_url = inp.get("video_url")
    language  = inp.get("language", "en")

    if not video_url:
        return {"ok": False, "error": "video_url required"}

    with tempfile.TemporaryDirectory() as tmp:
        raw      = os.path.join(tmp, "chunk.mp4")
        cropped  = os.path.join(tmp, "cropped.mp4")
        frames   = os.path.join(tmp, "frames")

        # 1 — Download
        try:
            logger.info("Downloading chunk...")
            _download(video_url, raw)
            logger.info("%.1f MB", os.path.getsize(raw) / 1e6)
        except Exception as e:
            return {"ok": False, "error": f"download: {e}"}

        # 2 — Crop to subtitle zone
        try:
            _crop_subtitle_zone(raw, cropped)
        except Exception as e:
            logger.warning("Crop failed (%s) — using full frame", e)
            cropped = raw

        # 3 — Extract frames at 2 fps
        try:
            _extract_frames(cropped, frames, fps=2)
        except Exception as e:
            return {"ok": False, "error": f"frame extraction: {e}"}

        # 4 — OCR
        try:
            rows = _ocr_frames(frames, language, fps=2)
        except Exception as e:
            return {"ok": False, "error": f"OCR: {e}"}

        # 5 — Build SRT
        srt = _rows_to_srt(rows, fps=2)
        cue_count = srt.count("\n\n")
        logger.info("Done: %d cues", cue_count)

        return {"ok": True, "srt": srt, "cue_count": cue_count}


runpod.serverless.start({"handler": handler})
