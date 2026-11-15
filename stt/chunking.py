"""Splitting long recordings.

Sending a 2 hour file to the model in one go is a bad idea: memory grows, one
failure throws away all the work, and hosted APIs have upload limits (Groq /
OpenAI cap around 25 MB). So long audio is cut into pieces of a few minutes.

Cutting at a fixed 5:00 mark would slice words in half, so I look for pauses
and cut in the middle of the nearest one instead.

Pauses come from Silero VAD (a small speech detector that ships with
faster-whisper). I first tried ffmpeg's silencedetect with a dB threshold, but
on old or noisy recordings (the JFK sample has crowd noise and tape hiss) the
"silence" is never quiet enough, so it found nothing. A VAD looks for speech,
not loudness, so it doesn't care about the noise floor. silencedetect is kept
as a fallback for setups without faster-whisper (e.g. Groq only).
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([\d.]+)")


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_silences(wav: str | Path, min_silence: float = 0.3) -> list[tuple[float, float]]:
    """Return (start, end) of every pause longer than `min_silence` seconds."""
    try:
        return _vad_silences(wav, min_silence)
    except ImportError:
        return _ffmpeg_silences(wav, min_silence=min_silence)


def _vad_silences(wav, min_silence):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    from .engines import read_wav

    sr = 16000
    speech = get_speech_timestamps(read_wav(wav), VadOptions(min_silence_duration_ms=int(min_silence * 1000)))
    # the gaps between speech regions are the pauses
    return [(a["end"] / sr, b["start"] / sr) for a, b in zip(speech, speech[1:])]


def _ffmpeg_silences(wav, noise_db: int = -35, min_silence: float = 0.4):
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-i", str(wav),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    log = subprocess.run(cmd, capture_output=True, text=True).stderr

    silences, current = [], None
    for line in log.splitlines():
        if m := _SILENCE_START.search(line):
            current = max(0.0, float(m.group(1)))
        elif (m := _SILENCE_END.search(line)) and current is not None:
            silences.append((current, float(m.group(1))))
            current = None
    return silences


def plan_chunks(duration: float, silences: list[tuple[float, float]],
                target: float = 240.0, max_len: float = 300.0) -> list[Chunk]:
    """Decide where to cut.

    Aim for `target` seconds per chunk; cut at the middle of the silence closest
    to that point, but never let a chunk grow past `max_len`. If there's no
    usable pause (music, nonstop talking) fall back to a hard cut at max_len.
    """
    if duration <= max_len:
        return [Chunk(0, 0.0, duration)]

    cut_points = [(s + e) / 2 for s, e in silences]
    chunks, start = [], 0.0

    while duration - start > max_len:
        lo, hi = start + target * 0.5, start + max_len
        candidates = [c for c in cut_points if lo <= c <= hi]
        if candidates:
            cut = min(candidates, key=lambda c: abs(c - (start + target)))
        else:
            cut = hi
        chunks.append(Chunk(len(chunks), start, cut))
        start = cut

    chunks.append(Chunk(len(chunks), start, duration))
    return chunks
