"""Ensures the project root is on sys.path so `import src...` works under pytest,
regardless of how pytest is invoked (`pytest` vs `python -m pytest`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Tests must never depend on a live, paid LLM API — that's slow, costs money,
# and (as discovered the hard way) can burn a free-tier rate limit before the
# tests that actually need to check LLM-specific behavior even run. The app is
# explicitly designed to work with zero credentials (extractive fallback +
# keyword grading), so the test suite always runs in that mode by default,
# regardless of what's in your local .env.
#
# To intentionally run the suite against a real LLM (e.g. to sanity-check a
# provider integration), set RUN_LIVE_LLM_TESTS=1 first.
if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
    os.environ["LLM_API_KEY"] = ""
