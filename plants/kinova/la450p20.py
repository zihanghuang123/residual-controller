"""Kinova, lookahead sweep: H=450, 20 preview points (stride=H//points)."""

from pathlib import Path

from plants.kinova.config import *  # noqa: F403

OUTPUT_DIR = PROJECT_ROOT / "outputs" / PLANT_NAME / Path(__file__).stem
PURE_RNN = {**PURE_RNN, "n_rollout": 450, "lookahead_points": 20}
PURE_RNN["lookahead_stride"] = PURE_RNN["n_rollout"] // PURE_RNN["lookahead_points"]
