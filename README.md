# Transcription pipeline

Takes an audio (or video) file, transcribes it, and returns the text with
start/end timestamps per segment. Works as a CLI script and as an HTTP API.

```
$ python transcribe.py samples/jfk.flac --format txt
[00:00:00 - 00:00:10] And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```

The bigger sample is JFK's "We choose to go to the Moon" speech at Rice
University (1962, public domain): 17:48 of old, noisy audio. On my machine
(Quadro P2000, 5 GB, Whisper `small`) the whole thing transcribes in about
50 seconds. Output is in [`examples/`](examples/) as JSON, SRT and plain text.

## How it works

```
upload ──> ffprobe ──> ffmpeg ──> split at pauses ──> Whisper per chunk ──> shift timestamps ──> JSON / SRT / VTT / txt
          (validate)  (16 kHz     (Silero VAD,         (checkpoint each        back onto the
                       mono WAV)   ~4 min chunks)       chunk to disk)          full timeline
```

| File | What it does |
|---|---|
| `stt/audio.py` | Probes the input and converts anything to 16 kHz mono WAV |
| `stt/chunking.py` | Finds pauses and decides where to cut long audio |
| `stt/engines.py` | STT backends: local faster-whisper, or Groq's hosted Whisper |
| `stt/pipeline.py` | Runs the steps above, saves each chunk's result, merges |
| `stt/formats.py` | SRT / VTT / txt output |
| `stt/api.py` | FastAPI service with sync endpoint + background jobs |
| `transcribe.py` | CLI |

## Running it

Needs Python 3.10+ and ffmpeg.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# NVIDIA GPU: also pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

# CLI
python transcribe.py samples/jfk_rice_moon_speech_1962.mp3 --format srt -o speech.srt
python transcribe.py some_video.mp4 --model medium --language en

# API (docs at http://localhost:8000/docs)
uvicorn stt.api:app --port 8000

curl -F file=@samples/jfk.flac localhost:8000/transcribe          # short file, waits for result
curl -F file=@samples/jfk_rice_moon_speech_1962.mp3 localhost:8000/jobs   # long file -> {"id": ...}
curl localhost:8000/jobs/<id>                                      # status, progress, result
curl "localhost:8000/jobs/<id>/export?format=srt"

# tests (no model download needed, they use a fake engine + ffmpeg generated audio)
pytest
```

To use Groq instead of the local model: `export GROQ_API_KEY=... STT_ENGINE=groq`
(or `--engine groq` on the CLI). The first local run downloads the Whisper
weights from Hugging Face (~500 MB for `small`).

Docker (CPU): `docker build -t stt . && docker run -p 8000:8000 -v $PWD/data:/data stt`

### Response format

```json
{
  "language": "en",
  "duration": 1068.07,
  "engine": "faster-whisper",
  "model": "small",
  "processing_seconds": 50.9,
  "source": {"file": "jfk_rice_moon_speech_1962.mp3", "format": "mp3", "codec": "mp3", "sample_rate": 44100, "channels": 1},
  "text": "President Pittman, Mr. Vice President, ...",
  "segments": [
    {"id": 0, "start": 2.9, "end": 12.52, "text": "President Pittman, Mr. Vice President, ...", "confidence": 0.749}
  ]
}
```

`confidence` is Whisper's average log probability for the segment mapped to
0..1. It's rough, but useful for flagging parts a human should double check.

## Design decisions

**Model: Whisper via faster-whisper.** I didn't want to train anything, the
task is about the pipeline. Whisper is the most accurate open model I know of
for messy real-world audio and it gives segment timestamps out of the box.
faster-whisper runs the same weights on CTranslate2, which is several times
faster than the original package and uses less memory (int8 on my older GPU).
It runs locally, so no per-minute cost and the audio never leaves the machine.

The model sits behind a tiny interface (`transcribe(wav) -> segments, language`),
so it can be swapped. I added Groq as a second backend because I already use
Groq's Whisper in a push-to-talk dictation tool on my own desktop, and for a
product it's a reasonable option when you don't want to run GPUs. Deepgram,
AssemblyAI or OpenAI would be one more class each.

**Formats: normalize once with ffmpeg.** MP3, WAV, M4A, FLAC, OGG, WebM, even
MP4 video: these are just containers and codecs around the same thing. Instead
of branching on file extensions, every input goes through `ffmpeg -vn -ac 1
-ar 16000` once, and the rest of the code only deals with one format (16 kHz
mono PCM, which is what Whisper wants anyway). Before converting, `ffprobe`
checks that the file really has an audio stream and a duration. The extension
and the client's MIME type are never trusted. Bad files fail fast with a 415
and a readable message.

**Long files: chunk at pauses, checkpoint each chunk.** Long audio is split
into ~4 minute chunks (max 5). Cutting at fixed times would slice words in
half, so it cuts in the middle of the pause closest to the target.

I started with ffmpeg's `silencedetect` for this, which worked on clean
audio but found zero pauses in the JFK recording: crowd noise and tape hiss
keep the "silent" parts above any sensible dB threshold. So pauses now come
from Silero VAD (bundled with faster-whisper), which detects speech rather
than loudness. `silencedetect` stays as a fallback.

Each chunk's timestamps are shifted by the chunk's start time so the final
output is on the full file's timeline. Every finished chunk is written to
disk (write to temp file, then rename) before moving on, so if the process
dies at chunk 7 of 10, running it again only does chunks 7 to 10. Smaller
chunks also keep memory flat and fit under hosted API upload limits (~25 MB).

**Sync vs async.** `/transcribe` is for short clips (default limit 2 min) and
just returns the result. Anything longer goes through `/jobs`: the upload is
streamed to disk, a job id comes back immediately with 202, and a background
worker processes it. Job state is a JSON file per job, so queued or half-done
jobs are picked up again after a restart. Transient errors (GPU OOM, network)
are retried with backoff up to 3 times; bad input is not retried.

## Scaling it up (Part 2)

What's here is one process with one worker thread, which is right for one GPU.
For a real service:

- **Concurrent uploads.** Keep the API stateless and cheap: clients upload
  straight to object storage with a presigned URL, the API creates a job row
  and pushes to a queue (Redis/Celery to start, SQS or RabbitMQ later). A pool
  of GPU workers pulls from the queue, each with the model already loaded.
  Scale workers on queue depth, not on HTTP traffic. A load balancer in front
  of the API instances handles the HTTP side; the queue is what absorbs a burst
  of 500 uploads. Add per-user rate limits, size limits, and a file hash so the
  same upload isn't transcribed twice.
- **Storage.** Audio in S3 / GCS / MinIO (original + normalized WAV), encrypted,
  with lifecycle rules to expire it. Transcripts and job metadata in Postgres:
  a `jobs` table and a `segments` table (job_id, start, end, text, confidence,
  speaker) so you can query by time range. Store the model version per job so
  you know what to re-run after a model upgrade. pgvector on segment embeddings
  if the transcripts feed search or RAG.
- **Failures.** The per-chunk checkpoints already make retries cheap. In
  production: workers ack the queue message only after the result is saved,
  so a crashed worker's job goes back to the queue; a visibility timeout
  catches stuck jobs; a dead-letter queue collects jobs that keep failing;
  alerts on failure rate.
- **API.** Versioned REST (`/v1/transcriptions`), API keys, webhooks when a
  job finishes so clients don't need to poll, SRT/VTT export, `DELETE` to
  remove audio + transcript, and a WebSocket endpoint if live transcription
  is ever needed.

## Limitations / what I'd do next

- No speaker diarization ("who said what"). pyannote or WhisperX would add it.
- Word-level timestamps are one flag away in faster-whisper (`word_timestamps=True`), not exposed yet.
- `/transcribe` and the job worker can use the GPU at the same time; fine for a demo, a real deployment would route everything through the queue.
- Small Whisper models occasionally mishear names ("thaw these mysteries" should be "solve"); `large-v3` fixes most of that at ~3x the cost.
