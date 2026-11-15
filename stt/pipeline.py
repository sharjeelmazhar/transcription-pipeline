"""The actual pipeline: probe -> normalize -> split -> transcribe chunks -> merge.

Each chunk's result is written to the job's work dir as soon as it's done. If
the process dies halfway through a long file (crash, OOM, deploy, power cut),
running the same job again skips the chunks that already have a result.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import audio
from .chunking import Chunk, find_silences, plan_chunks
from .engines import Segment

log = logging.getLogger("stt")


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment]
    engine: str = ""
    model: str = ""
    processing_seconds: float = 0.0
    source: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": round(self.duration, 2),
            "engine": self.engine,
            "model": self.model,
            "processing_seconds": round(self.processing_seconds, 2),
            "source": self.source,
            "text": self.text,
            "segments": [dict(id=i, **s.to_dict()) for i, s in enumerate(self.segments)],
        }


def transcribe_file(path: str | Path, engine, work_dir: str | Path,
                    language: str | None = None,
                    chunk_target: float = 240, chunk_max: float = 300,
                    on_progress: Callable[[int, int], None] | None = None) -> Transcript:
    t0 = time.monotonic()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    info = audio.probe(path)
    log.info("input %s: %s/%s %d Hz %d ch, %.1fs", Path(path).name, info.container,
             info.codec, info.sample_rate, info.channels, info.duration)

    wav = work / "audio.wav"
    if not wav.exists():
        audio.to_wav(path, wav)

    chunks = plan_chunks(info.duration, find_silences(wav) if info.duration > chunk_max else [],
                         target=chunk_target, max_len=chunk_max)
    log.info("%d chunk(s)", len(chunks))

    segments: list[Segment] = []
    detected = language
    for chunk in chunks:
        segs, lang = _run_chunk(chunk, wav, work, engine, language, single=len(chunks) == 1)
        detected = detected or lang
        segments.extend(segs)
        if on_progress:
            on_progress(chunk.index + 1, len(chunks))

    return Transcript(
        language=detected or "unknown",
        duration=info.duration,
        segments=segments,
        engine=engine.name,
        model=getattr(engine, "model_name", ""),
        processing_seconds=time.monotonic() - t0,
        source={"file": Path(path).name, "format": info.container, "codec": info.codec,
                "sample_rate": info.sample_rate, "channels": info.channels},
    )


def _run_chunk(chunk: Chunk, wav: Path, work: Path, engine, language, single: bool):
    done = work / f"chunk_{chunk.index:04d}.json"
    if done.exists():
        try:
            data = json.loads(done.read_text())
            log.info("chunk %d already done, skipping", chunk.index)
            return [Segment(**s) for s in data["segments"]], data["language"]
        except (json.JSONDecodeError, KeyError):
            log.warning("chunk %d result is corrupt, redoing it", chunk.index)

    if single:
        piece = wav
    else:
        piece = audio.to_wav(wav, work / f"chunk_{chunk.index:04d}.wav", chunk.start, chunk.end)

    log.info("chunk %d: %.1fs -> %.1fs", chunk.index, chunk.start, chunk.end)
    segs, lang = engine.transcribe(piece, language=language)
    # timestamps come back relative to the chunk, shift them back onto the full file
    segs = [s.shifted(chunk.start) for s in segs]

    tmp = done.with_suffix(".tmp")
    tmp.write_text(json.dumps({"language": lang, "segments": [s.to_dict() for s in segs]}))
    tmp.replace(done)
    if piece != wav:
        piece.unlink(missing_ok=True)
    return segs, lang
