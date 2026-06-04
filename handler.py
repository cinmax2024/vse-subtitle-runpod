"""
RunPod serverless handler for Video Subtitle Extractor (VSE).

Accepts a presigned R2 URL to a video chunk, runs VSE (PaddleOCR-based
OCR on video frames), and returns the extracted SRT as a plain string.

Expected input payload:
  {
    "video_url":    "<presigned R2 GET URL>",
    "language":     "en"   (optional, default "en")
    "crop_region":  {"x": 109, "y": 339, "w": 655, "h": 133}   (optional)
  }

Output:
  {"ok": true,  "srt": "<srt content>", "cue_count": N}
  {"ok": false, "error": "<message>"}

If crop_region is provided, the handler pre-crops the video with ffmpeg
before passing it to VSE. This skips auto-detection and excludes
logos/watermarks. Get coordinates from Shotcut → Filters → Spot Remover
(Position = x,y  and  Size = w,h).
"""
import glob
import logging
import os
import subprocess
import tempfile

import runpod

logger = logging.getLogger("vse_handler")
logging.basicConfig(level=logging.INFO)

VSE_DIR = os.environ.get("VSE_DIR", "/app/vse")


def _download(url, dest):
    result = subprocess.run(
        ["wget", "-q", "--timeout=120", "-O", dest, url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wget failed: {result.stderr[:500]}")


def _crop_video(input_path, output_path, crop):
    """Pre-crop video to subtitle region using ffmpeg."""
    x, y = crop["x"], crop["y"]
    w, h = crop["w"], crop["h"]
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-an", output_path,
    ], check=True, timeout=120)
    logger.info("Pre-cropped to %dx%d at (%d,%d)", w, h, x, y)


def handler(job):
    inp        = job.get("input", {})
    video_url  = inp.get("video_url")
    language   = inp.get("language", "en")
    crop       = inp.get("crop_region")

    if not video_url:
        return {"ok": False, "error": "video_url required"}

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "chunk.mp4")
        ocr_input  = video_path
        out_dir    = os.path.join(tmp, "out")
        os.makedirs(out_dir)

        # ── Download ──────────────────────────────────────────────
        try:
            logger.info("Downloading video chunk from R2...")
            _download(video_url, video_path)
        except Exception as exc:
            return {"ok": False, "error": f"download failed: {exc}"}

        size_mb = os.path.getsize(video_path) / 1024 / 1024
        logger.info("Chunk downloaded: %.1f MB", size_mb)

        # ── Pre-crop if coordinates supplied ──────────────────────
        if crop:
            cropped_path = os.path.join(tmp, "cropped.mp4")
            try:
                _crop_video(video_path, cropped_path, crop)
                ocr_input = cropped_path
            except Exception as exc:
                logger.warning("Crop failed (%s), falling back to full frame", exc)

        # ── Run VSE ───────────────────────────────────────────────
        try:
            result = subprocess.run(
                [
                    "python", "main.py",
                    "-i", ocr_input,
                    "-o", out_dir,
                    "--language", language,
                ],
                cwd=VSE_DIR,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.warning("VSE stderr: %s", result.stderr[:1000])
                return {
                    "ok": False,
                    "error": f"VSE exit {result.returncode}: {result.stderr[:500]}",
                }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "VSE timed out after 600s"}
        except Exception as exc:
            return {"ok": False, "error": f"VSE error: {exc}"}

        # ── Collect SRT output ────────────────────────────────────
        srts = (
            glob.glob(os.path.join(out_dir, "**", "*.srt"), recursive=True)
            or glob.glob(os.path.join(out_dir, "*.srt"))
        )
        if not srts:
            logger.warning("VSE stdout: %s", result.stdout[:500])
            return {"ok": False, "error": "VSE produced no .srt file"}

        srt_content = open(srts[0], encoding="utf-8", errors="replace").read()
        cue_count = srt_content.count("-->")
        logger.info("VSE complete: %d cues extracted", cue_count)
        return {"ok": True, "srt": srt_content, "cue_count": cue_count}


runpod.serverless.start({"handler": handler})
