"""Shared paths relative to the project directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "models"
CONFIG = PROJECT_ROOT / "config"
PERSONALITY_FILE = CONFIG / "jarvis_personality.txt"
CALIBRATION_FILE = CONFIG / "motion_calibration.json"
