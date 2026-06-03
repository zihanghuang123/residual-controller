"""Curriculum stage 1/5 for six_pendulum. Shared OUTPUT_DIR so subsequent stages warm-start."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from config import *  # noqa: F401, F403

PURE = {**PURE, "n_rollout": 500, "n_iterations": 3000}
OUTPUT_DIR = HERE.parent.parent / "outputs" / HERE.name / "config_curriculum"
