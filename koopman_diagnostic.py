"""
Coordinate-free discrete-symmetry check via the Koopman spectrum (PDF Phase 3).

Runs with no GPU and no training -- an early read on whether the data actually
carries a discrete symmetry that a coordinate map could linearize.

Usage:
    python3 create_synth.py z_dim=2 u_dim=64 timesteps=200 \
        train_initial_conds=512 val_initial_conds=20 test_initial_conds=100
    python3 koopman_diagnostic.py -DT synth -TS 200

    python3 create_lorenz.py train_initial_conds=256 val_initial_conds=20 test_initial_conds=20
    python3 koopman_diagnostic.py -DT lorenz -TS 250 --degree 2

Note: this script's own -DT/-TS/--degree/... flags are a small local argparse
parser (not cmd_line.parse_args), so they are unaffected by the Hydra
conversion and still use the old single-dash syntax.
"""

import argparse
import numpy as np

from src.utils.other import (get_synth_path, get_lorenz_path,
                             get_lorenz_distort_path)
from src.utils.koopman import koopman_diagnostic


CANDIDATES = {
    # dataset -> (candidate linear action S in canonical coords, finite order)
    "synth":          (np.diag([-1.0, -1.0]), 2),          # Duffing Z2 inversion
    "lorenz":         (np.diag([-1.0, -1.0, 1.0]), 2),     # Lorenz (x,y)->(-x,-y)
    "lorenz_distort": (np.diag([-1.0, -1.0, 1.0]), 2),     # distorted-Lorenz Z2
}

_PATHS = {"synth": get_synth_path, "lorenz": get_lorenz_path,
          "lorenz_distort": get_lorenz_distort_path}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-DT", "--data_set", default="synth", choices=list(CANDIDATES))
    ap.add_argument("-TS", "--timesteps", type=int, required=True,
                    help="steps per trajectory used when the dataset was generated")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--degree", type=int, default=3, help="Koopman dictionary degree")
    ap.add_argument("--pca_dim", type=int, default=6,
                    help="PCA dimension for the raw-observation EDMD")
    args = ap.parse_args()

    _, paths = _PATHS[args.data_set]()
    path = {"train": paths[0], "val": paths[1], "test": paths[2]}[args.split]
    data = np.load(path, allow_pickle=True).item()

    xdim, zdim = np.shape(data["x"])[-1], np.shape(data["z"])[-1]
    n = int(np.asarray(data["x"]).size // xdim)
    if n % args.timesteps != 0:
        raise SystemExit(f"{n} rows not divisible by -TS {args.timesteps}; "
                         f"pass the timesteps value used at generation time")
    print(f"loaded {path}: {n // args.timesteps} trajectories x {args.timesteps} steps, "
          f"x dim {xdim}, z dim {zdim}")

    S, order = CANDIDATES[args.data_set]
    koopman_diagnostic(data, timesteps=args.timesteps, candidate_S=S, order=order,
                       degree=args.degree, pca_dim=args.pca_dim)


if __name__ == "__main__":
    main()
