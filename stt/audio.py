"""Input handling: probe whatever the user uploaded and turn it into 16 kHz mono WAV.

Whisper (and pretty much every STT model) wants 16 kHz mono PCM, so instead of
special-casing mp3 / m4a / ogg / mp4 etc. everything goes through ffmpeg once
and the rest of the pipeline only ever sees one format.
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16000


class AudioError(Exception):
    """File is missing, corrupt, or has no audio stream."""


@dataclass
class AudioInfo:
    duration: float
    codec: str
    sample_rate: int
    channels: int
    container: str


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(f"{tool} not found on PATH (install ffmpeg)")
    return path


def probe(path: str | Path) -> AudioInfo:
    """Read the real stream info with ffprobe. We don't trust the file extension."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise AudioError(f"{path.name}: file is missing or empty")

    cmd = [
        _require("ffprobe"), "-v", "error",
        "-show_entries", "format=duration,format_name:stream=codec_type,codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # last line of ffprobe's error is the useful bit, without our internal paths
        reason = (proc.stderr.strip().splitlines() or ["unknown error"])[-1].split(": ")[-1]
        raise AudioError(f"{path.name}: not a readable audio file ({reason})")

    data = json.loads(proc.stdout or "{}")
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise AudioError(f"{path.name}: no audio stream found")

    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or 0)
    if duration <= 0:
        raise AudioError(f"{path.name}: could not determine duration")

    return AudioInfo(
        duration=duration,
        codec=audio.get("codec_name", "unknown"),
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        container=fmt.get("format_name", "unknown"),
    )


def to_wav(src: str | Path, dst: str | Path, start: float | None = None,
           end: float | None = None) -> Path:
    """Decode any input to 16 kHz mono 16-bit WAV, optionally only [start, end)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")

    cmd = [_require("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0):.3f}"]
    # -vn drops video (so .mp4/.webm uploads just work), -ac 1 downmixes, -ar resamples
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(tmp)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise AudioError(f"ffmpeg failed: {proc.stderr.strip()[:300]}")
    tmp.replace(dst)  # write-then-rename so a crash never leaves a half file behind
    return dst
