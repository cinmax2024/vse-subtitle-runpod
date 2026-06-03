"""
RunPod serverless handler for Video Subtitle Extractor (VSE).

Accepts a presigned R2 URL to a video chunk, runs VSE (PaddleOCR-based
OCR on video frames), and returns the extracted SRT as a plain string.

Expected input payload:
  {
    "video_url": "<presigned R2 GET URL>",
    "language":  "en"   (optional, default "en")
  }

Output:
  {"ok": true,  "srt": "<srt content>", "cue_count": N}
  {"ok": false, "error": "<message>"}
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


def handler(job):
    inp       = job.get("input", {})
    video_url = inp.get("video_url")
    language  = inp.get("language", "en")

    if not video_url:
        return {"ok": False, "error": "video_url required"}

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "chunk.mp4")
        out_dir    = os.path.join(tmp, "out")
        os.makedirs(out_dir)

        try:
            logger.info("Downloading video chunk from R2...")
            _download(video_url, video_path)
        except Exception as exc:
            return {"ok": False, "error": f"download failed: {exc}"}

        size_mb = os.path.getsize(video_path) / 1024 / 1024
        logger.info("Chunk downloaded: %.1f MB", size_mb)

        try:
            result = subprocess.run(
                [
                    "python", "main.py",
                    "-i", video_path,
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

        srts = (
            glob.glob(os.path.join(out_dir, "**", "*.srt"), recursive=True)
            or glob.glob(os.path.join(out_dir, "*.srt"))
        )
        if not srts:
            logger.warning("VSE stdout: %s", result.stdout[:500])
            return {"ok": False, "error": "VSE produced no .srt file"}

        srt_content = open(srts[0], encoding="utf-8", errors="replace").read()
        cue_count = len([l for l in srt_content.splitlines() if l.strip().isdigit()])
        logger.info("VSE complete: %d cues extracted", cue_count)

        return {"ok": True, "srt": srt_content, "cue_count": cue_count}


runpod.serverless.start({"handler": handler})
