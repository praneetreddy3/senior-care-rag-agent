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
        "POST", "/chat/stream", json={"query": "What rights does a PACE participant have?", "history": []}
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert '"type": "sources"' in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body


def test_eval_check_returns_pass_fail_shape():
    """Used by the n8n regression-monitoring workflow (see automation/README.md)."""
    r = client.get("/eval-check?sample_size=4")
    assert r.status_code == 200
    body = r.json()
    assert "pass" in body and isinstance(body["pass"], bool)
    assert "failures" in body and isinstance(body["failures"], list)
    metrics = body["metrics"]
    for key in ("sample_size", "hit_rate", "grounded_rate", "keyword_coverage"):
        assert key in metrics
