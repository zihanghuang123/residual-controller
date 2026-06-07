"""Four-link, GRU 256x256, BPTT horizon sweep: n_rollout=1000."""

from pathlib import Path

from plants.four_pendulum.config import *  # noqa: F403

OUTPUT_DIR = PROJECT_ROOT / "outputs" / PLANT_NAME / Path(__file__).stem
PURE_RNN = {**PURE_RNN, "hidden_sizes": (256, 256), "n_rollout": 1000, "n_iterations": 20000}
