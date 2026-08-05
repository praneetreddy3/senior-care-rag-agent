"""API tests using FastAPI's TestClient (offline, TF-IDF backend)."""
import os

os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_home_serves_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert "Senior Care" in r.text


def test_ask_returns_answer():
    r = client.post("/ask", json={"query": "Who is eligible for PACE?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert isinstance(body["citations"], list)


def test_chat_stream_emits_events():
    with client.stream(
        "POST", "/chat/stream", json={"query": "How can I prevent falls?", "history": []}
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert '"type": "sources"' in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body
