#!/usr/bin/env python3
"""Command line version of the pipeline.

    python transcribe.py samples/jfk.flac
    python transcribe.py talk.mp3 --format srt -o talk.srt
    python transcribe.py meeting.m4a --engine groq
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from stt import transcribe_file
from stt.audio import AudioError
from stt.engines import load_engine
from stt.formats import to_srt, to_txt, to_vtt


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser(description="Transcribe an audio/video file with timestamps")
    p.add_argument("file", type=Path)
    p.add_argument("--engine", choices=["whisper", "groq"], default=None,
                   help="default: $STT_ENGINE or whisper (local)")
    p.add_argument("--model", default=None, help="e.g. small, medium, large-v3 / whisper-large-v3-turbo")
    p.add_argument("--language", default=None, help="skip auto detection, e.g. en")
    p.add_argument("--format", choices=["json", "srt", "vtt", "txt", "text"], default="json")
    p.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    p.add_argument("--work-dir", type=Path, default=Path(".work"),
                   help="intermediate files; re-running the same file resumes from here")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("faster_whisper", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.file.is_file():
        print(f"error: {args.file} not found", file=sys.stderr)
        return 1

    engine = load_engine(args.engine, args.model)
    # same file + same model -> same work dir, which is what makes resume work
    work = args.work_dir / f"{file_hash(args.file)}-{engine.name}-{engine.model_name}"

    try:
        t = transcribe_file(args.file, engine, work, language=args.language,
                            on_progress=lambda d, n: logging.info("progress %d/%d", d, n))
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out = {
        "json": lambda: json.dumps(t.to_dict(), indent=2, ensure_ascii=False),
        "srt": lambda: to_srt(t),
        "vtt": lambda: to_vtt(t),
        "txt": lambda: to_txt(t),
        "text": lambda: to_txt(t, timestamps=False),
    }[args.format]()

    if args.output:
        args.output.write_text(out, encoding="utf-8")
        logging.info("wrote %s (%.1fs audio in %.1fs)", args.output, t.duration, t.processing_seconds)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
