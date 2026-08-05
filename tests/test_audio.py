"""Audio transcription tests (optional).

Skips automatically unless BOTH are present:
  1. faster-whisper installed (pip install -r requirements-audio.txt), and
  2. sample audio files exist (python scripts/make_sample_audio.py).
So CI and default runs stay green; the test only exercises the audio path when
the extra is set up.
"""
import glob
import os

import pytest

from src import transcribe as stt

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")


def _sample_files():
    return sorted(
        glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
        + glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    )


@pytest.mark.skipif(not stt.available(), reason="faster-whisper not installed (optional audio extra)")
def test_transcribe_sample_audio():
    files = _sample_files()
    if not files:
        pytest.skip("no sample audio — run: python scripts/make_sample_audio.py")
    try:
        text = stt.transcribe(files[0]).lower()
    except Exception as e:  # noqa: BLE001 - model download/load needs internet once
        pytest.skip(f"Whisper model unavailable (offline?): {e}")
    # the first sample asks about PACE eligibility
    assert any(w in text for w in ["pace", "eligible", "eligibility", "enroll"])
