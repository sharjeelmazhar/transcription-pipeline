"""Speech-to-text backends.

The pipeline only depends on `transcribe(wav) -> (segments, language)`, so the
model is swappable. Two are included:

  * WhisperEngine - faster-whisper running locally (default, free, private)
  * GroqEngine    - Whisper large-v3 hosted on Groq's API (very fast, needs a key)

Adding Deepgram / AssemblyAI / OpenAI would just be another class here.
"""

import math
import os
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx
import numpy as np


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None  # avg log prob turned into 0..1, when the backend gives it

    def shifted(self, offset: float) -> "Segment":
        return Segment(round(self.start + offset, 2), round(self.end + offset, 2),
                       self.text, self.confidence)

    def to_dict(self) -> dict:
        return asdict(self)


def read_wav(path: str | Path) -> np.ndarray:
    """16-bit mono WAV (what audio.to_wav writes) -> float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono wav")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _prob(avg_logprob: float | None) -> float | None:
    if avg_logprob is None:
        return None
    return round(min(1.0, math.exp(avg_logprob)), 3)


def _preload_cuda_libs() -> None:
    """ctranslate2 needs cuBLAS 12 + cuDNN 9. If they came from pip (nvidia-cublas-cu12,
    nvidia-cudnn-cu12) they aren't on the loader path, so load them by hand."""
    import ctypes
    import glob
    import site

    for sp in site.getsitepackages():
        for pattern in ("nvidia/cublas/lib/libcublas*.so.*", "nvidia/cudnn/lib/libcudnn*.so.*"):
            for lib in sorted(glob.glob(f"{sp}/{pattern}")):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class WhisperEngine:
    name = "faster-whisper"

    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "auto"):
        from faster_whisper import WhisperModel  # imported here so the API can start without it

        # loading takes a few seconds and a GB+ of memory, so do it once per process
        self.model_name = model
        _preload_cuda_libs()
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.device = self.model.model.device

    def transcribe(self, wav: str | Path, language: str | None = None) -> tuple[list[Segment], str]:
        # audio is already normalized by audio.to_wav, so hand the samples over directly
        segments, info = self.model.transcribe(
            read_wav(wav),
            language=language,
            beam_size=5,
            vad_filter=True,  # skips long silences, also cuts down on hallucinated text
            vad_parameters={"min_silence_duration_ms": 500},
        )
        out = [
            Segment(round(s.start, 2), round(s.end, 2), s.text.strip(), _prob(s.avg_logprob))
            for s in segments  # generator, the actual decoding happens while iterating
            if s.text.strip()
        ]
        return out, info.language


class GroqEngine:
    name = "groq"
    url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, model: str = "whisper-large-v3-turbo", api_key: str | None = None, timeout: float = 120):
        self.model_name = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {self.api_key}"})

    def transcribe(self, wav: str | Path, language: str | None = None) -> tuple[list[Segment], str]:
        # verbose_json is what gives back segments with start/end times
        data = {"model": self.model_name, "response_format": "verbose_json"}
        if language:
            data["language"] = language

        with open(wav, "rb") as f:
            resp = self.client.post(self.url, data=data, files={"file": (Path(wav).name, f, "audio/wav")})
        resp.raise_for_status()
        body = resp.json()

        segs = [
            Segment(round(s["start"], 2), round(s["end"], 2), s["text"].strip(), _prob(s.get("avg_logprob")))
            for s in body.get("segments", [])
            if s.get("text", "").strip()
        ]
        return segs, body.get("language", language or "unknown")


def load_engine(name: str | None = None, model: str | None = None):
    name = (name or os.environ.get("STT_ENGINE", "whisper")).lower()
    if name in ("whisper", "faster-whisper", "local"):
        return WhisperEngine(model or os.environ.get("WHISPER_MODEL", "small"))
    if name == "groq":
        return GroqEngine(model or os.environ.get("GROQ_MODEL", "whisper-large-v3-turbo"))
    raise ValueError(f"unknown engine: {name}")
