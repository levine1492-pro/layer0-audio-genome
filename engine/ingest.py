"""Audio ingestion, queue parsing, and canonical 44.1kHz stereo float32 decoding."""
import hashlib
import re
import subprocess
import urllib.request
from pathlib import Path
import gdown
import soundfile as sf
import numpy as np

CANONICAL_SR = 44100


def parse_queue(queue_path: Path):
    """Parse links_queue.txt. Supports: URL | title | seed_id | tags | notes"""
    if not queue_path.exists():
        return []
    entries = []
    for idx, raw in enumerate(queue_path.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        entries.append({
            "line_index": idx,
            "raw_line": raw,
            "source": parts[0],
            "title": parts[1] if len(parts) > 1 and parts[1] else "",
            "seed_id": parts[2] if len(parts) > 2 and parts[2] else "",
            "tags": parts[3] if len(parts) > 3 and parts[3] else "",
            "notes": parts[4] if len(parts) > 4 and parts[4] else "",
        })
    return entries


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()).strip("_")
    return s[:72] if s else "untitled_track"


def fetch_audio(source: str, work_dir: Path) -> Path:
    """Fetch audio from Google Drive, HTTP URL, or local path into work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    local_candidate = Path(source)
    if local_candidate.exists():
        return local_candidate

    if "drive.google.com" in source or "docs.google.com" in source:
        out = gdown.download(url=source, output=str(work_dir) + "/", quiet=False)
        if not out or not Path(out).exists():
            raise RuntimeError(f"gdown failed to fetch Google Drive file: {source}")
        return Path(out)

    if source.startswith("http://") or source.startswith("https://"):
        fname = source.split("?")[0].rstrip("/").split("/")[-1] or "downloaded_audio"
        dest = work_dir / fname
        urllib.request.urlretrieve(source, dest)
        return dest

    raise FileNotFoundError(f"Audio source not found: {source}")


def decode_canonical(raw_path: Path, work_dir: Path):
    """Transcode any audio/video container to 44.1kHz stereo float32 WAV."""
    sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()[:12]
    wav_path = work_dir / f"{sha}_canonical.wav"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(raw_path),
        "-vn", "-ac", "2", "-ar", str(CANONICAL_SR),
        "-c:a", "pcm_f32le", str(wav_path)
    ]
    subprocess.run(cmd, check=True)
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    stereo = data.T  # shape: (2, N)
    return stereo, sr, sha, raw_path.stem
