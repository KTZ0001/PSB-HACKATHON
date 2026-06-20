"""Pytest bootstrap: ensure the repo root is importable so `import src...` works."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
