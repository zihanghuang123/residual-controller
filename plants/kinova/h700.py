"""Kinova, GRU 256x256, BPTT horizon sweep: n_rollout=700."""

from pathlib import Path

from plants.kinova.config import *  # noqa: F403

OUTPUT_DIR = PROJECT_ROOT / "outputs" / PLANT_NAME / Path(__file__).stem
PURE_RNN = {**PURE_RNN, "n_rollout": 700}
