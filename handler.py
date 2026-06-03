"""
RunPod serverless handler — burned-in subtitle OCR via PaddleOCR.

Two-phase approach:
  Phase 1 — Auto-detect: scan 20 frames from the first 2 minutes,
             find the exact Y-band where subtitle text consistently lives.
  Phase 2 — Extract: only OCR frames where the image actually changed
             (keyframe-aware), cropped to the detected subtitle band.

No coordinates needed from the user — fully automatic.

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


# ── Timecode helper ───────────────────────────────────────────────────────────

def _tc(ms):
    ms = max(0, int(ms))
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Download ──────────────────────────────────────────────────────────────────

def _download(url, dest):
    r = subprocess.run(
        ["wget", "-q", "--timeout=120", "-O", dest, url],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"wget: {r.stderr[:400]}")


# ── Phase 1: auto-detect subtitle Y-band ─────────────────────────────────────

def _detect_subtitle_band(video_path, ocr, sample_count=20):
    """
    Scan `sample_count` evenly-spaced frames from the first 2 minutes.
    Run OCR on each frame and collect the Y positions of all detected text.
    Return (y_top, y_bottom) in pixels — the subtitle band.
    Falls back to bottom 20% if detection fails.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Sample from first 2 minutes max
    scan_frames = min(int(fps * 120), int(total))
    step = max(1, scan_frames // sample_count)

    y_hits = []  # Y-centre of each detected text box

    for i in range(0, scan_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue

        # Quick OCR on the bottom 40% only (save time during detection phase)
        top40 = int(height * 0.60)
        crop = frame[top40:, :]

        try:
            result = ocr.ocr(crop, cls=False)
        except Exception:
            continue
        if not result or not result[0]:
            continue

        for line in result[0]:
            if not line or len(line) < 2:
                continue
            conf = line[1][1]
            if conf < 0.65:
                continue
            # box is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] relative to crop
            box = line[0]
            y_centre = (box[0][1] + box[2][1]) / 2 + top40
            y_hits.append(y_centre)

    cap.release()

    if len(y_hits) < 3:
        # Not enough text found — fall back to bottom 18%
        logger.info("Detection: too few hits, using bottom 18%%")
        return int(height * 0.82), height

    y_arr = sorted(y_hits)
    # Find the densest cluster of Y values (where subtitles consistently appear)
    median_y = sorted(y_arr)[len(y_arr) // 2]
    band_y_top    = max(0,      int(median_y - height * 0.08))
    band_y_bottom = min(height, int(median_y + height * 0.06))

    logger.info(
        "Detected subtitle band: y=%d–%d (out of %d px height)",
        band_y_top, band_y_bottom, height,
    )
    return band_y_top, band_y_bottom


# ── Phase 2: keyframe-aware extraction ───────────────────────────────────────

def _frames_changed(prev, curr, threshold=0.08):
    """
    Return True if curr frame is different enough from prev to warrant OCR.
    Compares the mean absolute difference of the subtitle crop.
    """
    import numpy as np
    if prev is None:
        return True
    diff = abs(curr.astype(float) - prev.astype(float)).mean() / 255.0
    return diff > threshold


def _extract_and_ocr(video_path, ocr, y_top, y_bottom, language):
    """
    Walk through every frame, only run OCR when the subtitle zone changes.
    Returns list of (timestamp_ms, text).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Processing %d frames at %.1f fps", total, fps)

    rows     = []
    prev_crop = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = int(frame_idx / fps * 1000)
        crop  = frame[y_top:y_bottom, :]

        if _frames_changed(prev_crop, crop):
            try:
                result = ocr.ocr(crop, cls=False)
            except Exception:
                result = None

            if result and result[0]:
                parts = []
                for line in result[0]:
                    if not line or len(line) < 2:
                        continue
                    text, conf = line[1][0], line[1][1]
                    if conf >= 0.60 and len(text.strip()) >= 2:
                        parts.append(text.strip())
                if parts:
                    rows.append((ts_ms, " ".join(parts)))

            prev_crop = crop.copy()

        frame_idx += 1

    cap.release()
    logger.info("OCR complete: %d text events from %d frames", len(rows), total)
    return rows


# ── Build SRT from text events ────────────────────────────────────────────────

def _rows_to_srt(rows, fps=25):
    """Merge consecutive identical text into SRT cues."""
    if not rows:
        return ""

    frame_ms = max(1, int(1000 / fps))
    cues = []
    cur_text  = rows[0][1]
    cur_start = rows[0][0]
    cur_end   = rows[0][0] + frame_ms

    for ts, text in rows[1:]:
        if text == cur_text and ts <= cur_end + frame_ms * 4:
            cur_end = ts + frame_ms
        else:
            if cur_end - cur_start >= 300:
                cues.append((cur_start, cur_end, cur_text))
            cur_text  = text
            cur_start = ts
            cur_end   = ts + frame_ms

    if cur_end - cur_start >= 300:
        cues.append((cur_start, cur_end, cur_text))

    lines = []
    for i, (s, e, t) in enumerate(cues, 1):
        lines += [str(i), f"{_tc(s)} --> {_tc(e)}", t, ""]
    return "\n".join(lines)


# ── RunPod handler ────────────────────────────────────────────────────────────

def handler(job):
    inp       = job.get("input", {})
    video_url = inp.get("video_url")
    language  = inp.get("language", "en")

    if not video_url:
        return {"ok": False, "error": "video_url required"}

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "chunk.mp4")

        # Download
        try:
            logger.info("Downloading chunk...")
            _download(video_url, video_path)
            logger.info("%.1f MB downloaded", os.path.getsize(video_path) / 1e6)
        except Exception as e:
            return {"ok": False, "error": f"download: {e}"}

        # Init PaddleOCR once (GPU)
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=False, lang=language,
                            show_log=False, use_gpu=True)
        except Exception as e:
            return {"ok": False, "error": f"PaddleOCR init: {e}"}

        # Phase 1 — auto-detect subtitle band
        try:
            y_top, y_bottom = _detect_subtitle_band(video_path, ocr)
        except Exception as e:
            logger.warning("Band detection failed (%s) — using bottom 18%%", e)
            import cv2
            cap = cv2.VideoCapture(video_path)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            cap.release()
            y_top, y_bottom = int(h * 0.82), h

        # Phase 2 — keyframe-aware extraction
        try:
            rows = _extract_and_ocr(video_path, ocr, y_top, y_bottom, language)
        except Exception as e:
            return {"ok": False, "error": f"extraction: {e}"}

        # Build SRT
        srt = _rows_to_srt(rows)
        cue_count = srt.count("\n\n")
        logger.info("Result: %d subtitle cues", cue_count)

        return {"ok": True, "srt": srt, "cue_count": cue_count}


runpod.serverless.start({"handler": handler})
