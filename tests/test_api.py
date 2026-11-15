import time

import pytest
from fastapi.testclient import TestClient

from tests.test_pipeline import FakeEngine, make_audio


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STT_DATA_DIR", str(tmp_path / "data"))
    import importlib
    import stt.api as api
    api = importlib.reload(api)
    monkeypatch.setattr(api, "load_engine", lambda: FakeEngine())
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_sync_transcribe(client, tmp_path):
    f = make_audio(tmp_path / "a.mp3", [("tone", 3)])
    r = client.post("/transcribe", files={"file": ("a.mp3", f.read_bytes(), "audio/mpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "first half second half"
    assert body["segments"][1]["start"] == pytest.approx(1.5, abs=0.1)


def test_bad_upload_is_415(client):
    r = client.post("/transcribe", files={"file": ("x.mp3", b"not audio", "audio/mpeg")})
    assert r.status_code == 415


def test_job_flow(client, tmp_path):
    f = make_audio(tmp_path / "b.wav", [("tone", 4)])
    r = client.post("/jobs", files={"file": ("b.wav", f.read_bytes(), "audio/wav")})
    assert r.status_code == 202
    job_id = r.json()["id"]

    for _ in range(50):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] == "completed":
            break
        time.sleep(0.1)
    assert job["status"] == "completed"
    assert job["result"]["segments"]

    srt = client.get(f"/jobs/{job_id}/export", params={"format": "srt"}).text
    assert "-->" in srt


def test_unknown_job(client):
    assert client.get("/jobs/nope").status_code == 404
