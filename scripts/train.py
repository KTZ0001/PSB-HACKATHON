"""Train the Aegis behavioral risk engine. Thin CLI wrapper.

Usage:  python scripts/train.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.train import train_models

if __name__ == "__main__":
    train_models()
