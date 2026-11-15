"""HTTP API.

    POST /transcribe          short files, waits and returns the transcript
    POST /jobs                any length, returns a job id immediately (202)
    GET  /jobs/{id}           status + progress, and the result when it's done
    GET  /jobs/{id}/export    ?format=srt|vtt|txt|json
    GET  /health

Run:  uvicorn stt.api:app --port 8000   (docs at /docs)

Jobs run on a single background worker thread because there's one model on one
GPU. In production this would be a real queue (Redis/SQS) with a pool of GPU
workers, see the README.
"""

import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .audio import AudioError, probe
from .engines import Segment, load_engine
from .formats import to_srt, to_txt, to_vtt
from .pipeline import Transcript, transcribe_file

log = logging.getLogger("stt.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

DATA_DIR = Path(os.environ.get("STT_DATA_DIR", "data"))
MAX_UPLOAD_MB = int(os.environ.get("STT_MAX_UPLOAD_MB", "500"))
SYNC_LIMIT_SECONDS = float(os.environ.get("STT_SYNC_LIMIT", "120"))
MAX_ATTEMPTS = 3

state: dict = {"engine": None, "queue": queue.Queue(), "jobs": {}}
jobs_lock = threading.Lock()


# --- job bookkeeping (a JSON file per job, so jobs survive a restart) --------

def _job_dir(job_id: str) -> Path:
    return DATA_DIR / "jobs" / job_id


def _save_job(job: dict) -> None:
    path = _job_dir(job["id"]) / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2))
    tmp.replace(path)


def _update(job_id: str, **fields) -> dict:
    with jobs_lock:
        job = state["jobs"][job_id]
        job.update(fields, updated_at=time.time())
        _save_job(job)
        return job


def _load_jobs() -> None:
    """On startup pick up whatever was queued or running when the server stopped."""
    for path in sorted((DATA_DIR / "jobs").glob("*/job.json")):
        job = json.loads(path.read_text())
        state["jobs"][job["id"]] = job
        if job["status"] in ("queued", "processing"):
            job["status"] = "queued"
            _save_job(job)
            state["queue"].put(job["id"])
            log.info("re-queued %s after restart", job["id"])


# --- worker ------------------------------------------------------------------

def _worker() -> None:
    while True:
        job_id = state["queue"].get()
        job = state["jobs"][job_id]
        _update(job_id, status="processing", attempts=job["attempts"] + 1)
        try:
            t = transcribe_file(
                job["input"], state["engine"], _job_dir(job_id) / "work",
                language=job.get("language"),
                on_progress=lambda d, n: _update(job_id, progress={"done": d, "total": n}),
            )
            t.source["file"] = job["filename"]
            (_job_dir(job_id) / "result.json").write_text(json.dumps(t.to_dict(), ensure_ascii=False))
            _update(job_id, status="completed", error=None)
        except AudioError as e:
            # bad input won't get better by retrying
            _update(job_id, status="failed", error=str(e))
        except Exception as e:  # GPU OOM, network error to Groq, etc.
            log.exception("job %s failed", job_id)
            attempts = state["jobs"][job_id]["attempts"]
            if attempts < MAX_ATTEMPTS:
                delay = 2 ** attempts
                _update(job_id, status="queued", error=f"retrying in {delay}s: {e}")
                threading.Timer(delay, state["queue"].put, args=(job_id,)).start()
            else:
                _update(job_id, status="failed", error=str(e))
        finally:
            state["queue"].task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    (DATA_DIR / "jobs").mkdir(parents=True, exist_ok=True)
    state["engine"] = load_engine()
    _load_jobs()
    threading.Thread(target=_worker, daemon=True, name="stt-worker").start()
    yield


app = FastAPI(title="Transcription service", version="1.0", lifespan=lifespan)


# --- helpers -----------------------------------------------------------------

async def _store_upload(file: UploadFile, dest_dir: Path) -> Path:
    """Stream the upload to disk in 1 MB pieces instead of reading it all into RAM."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload").suffix.lower()[:10]
    dest = dest_dir / f"input{suffix}"
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(dest, "wb") as out:
        while block := await file.read(1 << 20):
            size += len(block)
            if size > limit:
                out.close()
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")
            out.write(block)
    return dest


def _check_audio(path: Path, cleanup: Path):
    try:
        return probe(path)
    except AudioError as e:
        shutil.rmtree(cleanup, ignore_errors=True)
        raise HTTPException(415, str(e))


def _load_result(job_id: str) -> Transcript:
    data = json.loads((_job_dir(job_id) / "result.json").read_text())
    segs = [Segment(s["start"], s["end"], s["text"], s.get("confidence")) for s in data["segments"]]
    return Transcript(data["language"], data["duration"], segs, data["engine"], data["model"],
                      data["processing_seconds"], data["source"])


# --- routes ------------------------------------------------------------------

@app.get("/health")
def health():
    engine = state["engine"]
    return {"status": "ok", "engine": engine.name, "model": engine.model_name,
            "queued": state["queue"].qsize()}


@app.post("/transcribe")
async def transcribe_now(file: UploadFile = File(...), language: str | None = Form(None)):
    tmp_dir = DATA_DIR / "tmp" / uuid.uuid4().hex
    path = await _store_upload(file, tmp_dir)
    info = _check_audio(path, tmp_dir)
    if info.duration > SYNC_LIMIT_SECONDS:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(413, f"audio is {info.duration:.0f}s, use POST /jobs for anything "
                                 f"longer than {SYNC_LIMIT_SECONDS:.0f}s")
    try:
        # note: runs next to the background worker, fine for a demo with one GPU
        t = transcribe_file(path, state["engine"], tmp_dir / "work", language=language)
        t.source["file"] = file.filename
        return t.to_dict()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/jobs", status_code=202)
async def create_job(file: UploadFile = File(...), language: str | None = Form(None)):
    job_id = uuid.uuid4().hex[:12]
    path = await _store_upload(file, _job_dir(job_id))
    info = _check_audio(path, _job_dir(job_id))

    job = {
        "id": job_id, "status": "queued", "filename": file.filename, "input": str(path),
        "duration": round(info.duration, 2), "language": language, "attempts": 0,
        "progress": {"done": 0, "total": None}, "error": None,
        "created_at": time.time(), "updated_at": time.time(),
    }
    with jobs_lock:
        state["jobs"][job_id] = job
        _save_job(job)
    state["queue"].put(job_id)
    return {"id": job_id, "status": "queued", "duration": job["duration"]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = state["jobs"].get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    public = {k: v for k, v in job.items() if k != "input"}
    if job["status"] == "completed":
        public["result"] = _load_result(job_id).to_dict()
    return public


@app.get("/jobs/{job_id}/export")
def export_job(job_id: str, format: str = "srt"):
    job = state["jobs"].get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] != "completed":
        raise HTTPException(409, f"job is {job['status']}")
    t = _load_result(job_id)
    render = {"srt": to_srt, "vtt": to_vtt, "txt": to_txt,
              "json": lambda t: json.dumps(t.to_dict(), indent=2, ensure_ascii=False)}
    if format not in render:
        raise HTTPException(400, "format must be srt, vtt, txt or json")
    return PlainTextResponse(render[format](t))
