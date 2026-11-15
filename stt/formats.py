"""Output formats. JSON is the main one, SRT/VTT for subtitles, txt for humans."""

from .pipeline import Transcript


def _ts(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(t: Transcript) -> str:
    blocks = []
    for i, s in enumerate(t.segments, 1):
        blocks.append(f"{i}\n{_ts(s.start)} --> {_ts(s.end)}\n{s.text}\n")
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for s in t.segments:
        lines += [f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}", s.text, ""]
    return "\n".join(lines)


def to_txt(t: Transcript, timestamps: bool = True) -> str:
    if not timestamps:
        return t.text + "\n"
    return "\n".join(f"[{_ts(s.start, '.')[:-4]} - {_ts(s.end, '.')[:-4]}] {s.text}"
                     for s in t.segments) + "\n"
