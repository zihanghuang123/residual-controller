"""Six-link pendulum variant 1: inherits config.py, overrides GRU hidden size + rollout length."""

from pathlib import Path

from plants.six_pendulum.config import *  # noqa: F403

OUTPUT_DIR = PROJECT_ROOT / "outputs" / PLANT_NAME / Path(__file__).stem  # own output dir
PURE_RNN = {**PURE_RNN, "hidden_sizes": (256, 256), "n_rollout": 1000}
