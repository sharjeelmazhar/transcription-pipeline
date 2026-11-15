"""Tests that don't need the real model: a fake engine stands in for Whisper.

ffmpeg generates the test audio (tones with gaps of silence), so nothing has
to be downloaded.
"""

import json
import subprocess

import pytest

from stt import audio
from stt.chunking import _ffmpeg_silences, plan_chunks
from stt.engines import Segment
from stt.formats import to_srt
from stt.pipeline import transcribe_file


class FakeEngine:
    name = "fake"
    model_name = "fake"

    def __init__(self):
        self.calls = []

    def transcribe(self, wav, language=None):
        dur = audio.probe(wav).duration
        self.calls.append(dur)
        return [Segment(0.0, round(dur / 2, 2), "first half"),
                Segment(round(dur / 2, 2), round(dur, 2), "second half")], "en"


def make_audio(path, pattern, ext_args=()):
    """pattern: list of ('tone'|'silence', seconds)."""
    inputs, labels = [], []
    for i, (kind, secs) in enumerate(pattern):
        src = f"sine=frequency=440:duration={secs}" if kind == "tone" else f"anullsrc=r=44100:cl=stereo:d={secs}"
        inputs += ["-f", "lavfi", "-i", src]
        labels.append(f"[{i}:a]")
    graph = "".join(labels) + f"concat=n={len(pattern)}:v=0:a=1[out]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
                    "-map", "[out]", "-ac", "2", *ext_args, str(path)], check=True)
    return path


@pytest.mark.parametrize("ext", ["mp3", "wav", "ogg", "flac", "m4a"])
def test_any_format_becomes_16k_mono(tmp_path, ext):
    src = make_audio(tmp_path / f"in.{ext}", [("tone", 2)])
    out = audio.to_wav(src, tmp_path / "out.wav")
    info = audio.probe(out)
    assert (info.sample_rate, info.channels, info.codec) == (16000, 1, "pcm_s16le")
    assert info.duration == pytest.approx(2, abs=0.1)


def test_video_file_audio_is_extracted(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video)], check=True)
    info = audio.probe(audio.to_wav(video, tmp_path / "out.wav"))
    assert info.channels == 1 and info.sample_rate == 16000


def test_rejects_non_audio(tmp_path):
    junk = tmp_path / "notes.mp3"
    junk.write_text("definitely not audio")
    with pytest.raises(audio.AudioError):
        audio.probe(junk)

    empty = tmp_path / "empty.wav"
    empty.touch()
    with pytest.raises(audio.AudioError):
        audio.probe(empty)


def test_short_audio_is_one_chunk():
    assert len(plan_chunks(100, [], max_len=300)) == 1


def test_cuts_land_in_silence():
    # pauses every ~50s, chunks should end inside one of them
    silences = [(t, t + 1.0) for t in range(50, 1000, 50)]
    chunks = plan_chunks(1000, silences, target=240, max_len=300)
    assert chunks[0].start == 0 and chunks[-1].end == 1000
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == b.start
    for c in chunks[:-1]:
        assert any(s <= c.end <= e for s, e in silences)
        assert c.duration <= 300


def test_falls_back_to_hard_cut_without_silence():
    chunks = plan_chunks(700, [], target=240, max_len=300)
    assert [c.duration for c in chunks] == [300, 300, 100]


def test_ffmpeg_silence_fallback(tmp_path):
    # pure tones aren't speech, so this checks the silencedetect fallback path
    wav = make_audio(tmp_path / "gaps.wav", [("tone", 3), ("silence", 1.5), ("tone", 3)])
    (start, end), = _ffmpeg_silences(wav)
    assert start == pytest.approx(3, abs=0.1) and end == pytest.approx(4.5, abs=0.1)


def test_long_file_timestamps_are_offset(tmp_path):
    wav = make_audio(tmp_path / "long.wav", [("tone", 20), ("silence", 1), ("tone", 20)])
    engine = FakeEngine()
    t = transcribe_file(wav, engine, tmp_path / "work", chunk_target=15, chunk_max=25)

    assert len(engine.calls) == 2
    starts = [s.start for s in t.segments]
    assert starts == sorted(starts)
    second_chunk_first_seg = t.segments[2]
    assert second_chunk_first_seg.start > 19  # shifted onto the full timeline, not 0.0
    assert t.segments[-1].end == pytest.approx(41, abs=0.2)


def test_resume_skips_finished_chunks(tmp_path):
    wav = make_audio(tmp_path / "long.wav", [("tone", 20), ("silence", 1), ("tone", 20)])
    work = tmp_path / "work"
    transcribe_file(wav, FakeEngine(), work, chunk_target=15, chunk_max=25)

    # pretend the process died before chunk 1 was saved
    (work / "chunk_0001.json").unlink()
    engine = FakeEngine()
    t = transcribe_file(wav, engine, work, chunk_target=15, chunk_max=25)
    assert len(engine.calls) == 1
    assert len(t.segments) == 4


def test_corrupt_checkpoint_is_redone(tmp_path):
    wav = make_audio(tmp_path / "a.wav", [("tone", 3)])
    work = tmp_path / "work"
    transcribe_file(wav, FakeEngine(), work)
    (work / "chunk_0000.json").write_text("{half written")
    engine = FakeEngine()
    transcribe_file(wav, engine, work)
    assert len(engine.calls) == 1
    assert json.loads((work / "chunk_0000.json").read_text())["language"] == "en"


def test_srt_output():
    from stt.pipeline import Transcript
    t = Transcript("en", 65.5, [Segment(0, 2.5, "Hello."), Segment(61.25, 65.5, "Bye.")])
    assert to_srt(t) == ("1\n00:00:00,000 --> 00:00:02,500\nHello.\n\n"
                         "2\n00:01:01,250 --> 00:01:05,500\nBye.\n")
