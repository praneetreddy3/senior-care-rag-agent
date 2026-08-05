"""Optional server-side speech-to-text using faster-whisper (open-source Whisper).

This is separate from the browser's live-microphone Web Speech API: it lets the
server transcribe uploaded audio files, which also makes the voice pipeline
testable with audio fixtures.

Enable it with:  pip install -r requirements-audio.txt
The Whisper model downloads automatically on first use (needs internet once).
"""
from __future__ import annotations

import os
from functools import lru_cache

MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")  # tiny/base/small/... (tiny is fast, ~75MB)


def available() -> bool:
    """True if faster-whisper is installed (the optional audio extra)."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    # int8 on CPU keeps it light enough to run without a GPU.
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file (wav/mp3/m4a/...) to text."""
    segments, _info = _model().transcribe(audio_path)
    return "".join(seg.text for seg in segments).strip()
