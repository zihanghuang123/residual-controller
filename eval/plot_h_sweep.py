"""Box plots of endpoint + tracking-rms across a BPTT-horizon sweep (one box per H config).

Set PLANT once and list the config stems; reads each config's saved metrics_pure_rnn.npz,
running evaluate_pure_rnn for any config that lacks it.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLANT = "four_pendulum"
CONFIGS = ["h700", "h800", "h900", "h1000", "h1100", "h1200"]


def load_cfg(path):
    spec = importlib.util.spec_from_file_location("plant_cfg", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    entries = []  # (H, endpoint, tracking_rms)
    for stem in CONFIGS:
        config_path = PROJECT_ROOT / "plants" / PLANT / f"{stem}.py"
        cfg = load_cfg(config_path)
        npz = cfg.OUTPUT_DIR / "metrics_pure_rnn.npz"
        if not npz.exists():
            print(f"{stem}: no metrics, running evaluate_pure_rnn ...")
            subprocess.run([sys.executable, str(PROJECT_ROOT / "eval" / "evaluate_pure_rnn.py"),
                            "--config", str(config_path)], cwd=str(PROJECT_ROOT), check=False)
        if not npz.exists():
            print(f"skip {stem}: eval produced no {npz} (trained params missing?)")
            continue
        d = np.load(npz)
        entries.append((cfg.PURE_RNN["n_rollout"], d["endpoint_pure_rnn"], np.sqrt(d["tracking_pure_rnn"])))

    if not entries:
        print("no metrics found; run evaluate_pure_rnn.py on the sweep configs first")
        return

    entries.sort(key=lambda e: e[0])
    Hs = [e[0] for e in entries]
    eps = [e[1] for e in entries]
    trs = [e[2] for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(1.2 * len(Hs) + 3, 4))
    for ax, data, ylab in zip(axes, (eps, trs), ("endpoint error", "tracking rms")):
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(Hs) + 1))
        ax.set_xticklabels([str(h) for h in Hs])
        ax.set_xlabel("BPTT horizon H")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{PLANT}: pure_rnn over horizon sweep ({len(Hs)} configs)")
    fig.tight_layout()
    out = PROJECT_ROOT / "outputs" / PLANT / "h_sweep_box.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
