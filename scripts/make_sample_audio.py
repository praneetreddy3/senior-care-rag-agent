"""Generate sample spoken-question audio files for testing /transcribe.

Requires the audio extra and internet (gTTS):
    pip install -r requirements-audio.txt
    python scripts/make_sample_audio.py

Writes .mp3 files into tests/audio/, which tests/test_audio.py then transcribes.
"""
import os

from gtts import gTTS

QUESTIONS = {
    "q1_pace_eligibility": "Who is eligible for PACE?",
    "q2_fall_prevention": "How can I prevent falls at home?",
    "q3_medication": "What is medication reconciliation?",
}

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "audio")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, text in QUESTIONS.items():
        path = os.path.abspath(os.path.join(OUT, name + ".mp3"))
        gTTS(text).save(path)
        print("wrote", path)
    print("Done. Now run: python -m pytest tests/test_audio.py -q")


if __name__ == "__main__":
    main()
