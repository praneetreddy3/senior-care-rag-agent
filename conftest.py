"""Ensures the project root is on sys.path so `import src...` works under pytest,
regardless of how pytest is invoked (`pytest` vs `python -m pytest`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
