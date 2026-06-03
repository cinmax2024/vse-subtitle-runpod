"""
RunPod serverless handler for Video Subtitle Extractor (VSE).

Key design decisions:
- Pre-crops the video to the subtitle zone (bottom 22% of frame) BEFORE
  running OCR. This eliminates channel logos (TRT 1), watermarks (ForumKa),
  episode titles, and "Directed by" credits that appear outside the sub zone.
- Calls backend/main.py (headless), not main.py (PySide6 GUI).

Input payload:
  {
    "video_url":      "<presigned R2 GET URL>",
    "language":       "en"    (optional, default "en"),
    "sub_area_ratio": 0.22    (optional, fraction of frame height from bottom)
  }

Output:
  {"ok": true,  "srt": "<srt content>", "cue_count": N}
  {"ok": false, "error": "<message>"}
"""

import glob
import json
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


def _crop_to_subtitle_zone(src, dest, ratio):
    """
    Crop video to the bottom `ratio` fraction of the frame.
    e.g. ratio=0.22 → bottom 22% only (where subtitles live).
    Removes top-corner logos, watermarks, episode title cards.
    Audio is stream-copied unchanged.
    """
    start = 1.0 - ratio          # e.g. 0.78 — top of the crop window
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-vf", f"crop=in_w:in_h*{ratio}:0:in_h*{start}",
        "-c:a", "copy",
        dest,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg crop failed: {result.stderr[:400]}")


def handler(job):
    inp            = job.get("input", {})
    video_url      = inp.get("video_url")
    language       = inp.get("language", "en")
    sub_area_ratio = float(inp.get("sub_area_ratio", 0.22))

    if not video_url:
        return {"ok": False, "error": "video_url required"}

    with tempfile.TemporaryDirectory() as tmp:
        raw_path     = os.path.join(tmp, "chunk.mp4")
        cropped_path = os.path.join(tmp, "chunk_cropped.mp4")
        out_dir      = os.path.join(tmp, "out")
        os.makedirs(out_dir)

        # 1 — Download chunk from R2
        try:
            logger.info("Downloading chunk...")
            _download(video_url, raw_path)
        except Exception as exc:
            return {"ok": False, "error": f"download failed: {exc}"}

        size_mb = os.path.getsize(raw_path) / 1024 / 1024
        logger.info("Downloaded %.1f MB", size_mb)

        # 2 — Crop to subtitle zone (removes logos/watermarks outside the zone)
        try:
            logger.info("Cropping to bottom %.0f%% of frame...", sub_area_ratio * 100)
            _crop_to_subtitle_zone(raw_path, cropped_path, sub_area_ratio)
        except Exception as exc:
            logger.warning("Crop failed (%s) — using full frame", exc)
            cropped_path = raw_path   # fallback: use uncropped

        # 3 — Run VSE headless (backend/main.py, not main.py which is the PySide6 GUI)
        try:
            result = subprocess.run(
                [
                    "python", "backend/main.py",
                    "-i", cropped_path,
                    "-o", out_dir,
                    "--language", language,
                ],
                cwd=VSE_DIR,
                capture_output=True,
                text=True,
                timeout=840,
            )
            if result.returncode != 0:
                logger.warning("VSE stderr: %s", result.stderr[:1000])
                return {
                    "ok": False,
                    "error": f"VSE exit {result.returncode}: {result.stderr[:500]}",
                }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "VSE timed out after 840s"}
        except Exception as exc:
            return {"ok": False, "error": f"VSE error: {exc}"}

        # 4 — Collect SRT output
        srts = (
            glob.glob(os.path.join(out_dir, "**", "*.srt"), recursive=True)
            or glob.glob(os.path.join(out_dir, "*.srt"))
        )
        if not srts:
            logger.warning("VSE stdout: %s", result.stdout[:500])
            return {"ok": False, "error": "VSE produced no .srt file"}

        srt_content = open(srts[0], encoding="utf-8", errors="replace").read()
        cue_count   = len([l for l in srt_content.splitlines() if l.strip().isdigit()])
        logger.info("VSE complete: %d cues", cue_count)

        return {"ok": True, "srt": srt_content, "cue_count": cue_count}


runpod.serverless.start({"handler": handler})
