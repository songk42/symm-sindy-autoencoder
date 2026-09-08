"""
Combine per-trajectory cylinder windows (data/cylinder/_raw/{role}_seed*.npy,
written by create_cylinder.py) into the final data/cylinder*/{train,val,test}.npy
split files, optionally adding synthetic observation noise.

Pure numpy -- runs in the normal symm-sindy-autoencoder .venv, NOT hg-firedrake.

Noise convention matches the rest of the repo (-NSTR/--noise_strength on
synth/lorenz, src/dataset/{synth,lorenz}_functions.py): independent additive
Gaussian noise on x and dx separately (not re-differentiated), so a sweep here
is directly comparable to noise_strength sweeps on the synthetic testbeds.
Since the raw probe magnitude is O(0.01-0.1) (real pressure-coefficient
fluctuations, not unit-normalized), --noise_strength is an ABSOLUTE scale --
this script prints noise/signal RMS so you can read off the effective SNR.

Usage:
    # clean dataset
    python3 combine_cylinder.py --out_dir data/cylinder

    # a noise sweep, each level in its own dataset folder so cmd_line.py's
    # data_set= can point at whichever level you want to train on
    for n in 0.0 0.005 0.01 0.02 0.05; do
        python3 combine_cylinder.py --noise_strength $n \
            --out_dir data/cylinder_noise${n} --noise_seed 0
    done
"""

import argparse
import glob
import os

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default="data/cylinder/_raw",
                    help="directory of {role}_seed*.npy files from create_cylinder.py")
    ap.add_argument("--out_dir", default="data/cylinder")
    ap.add_argument("--noise_strength", type=float, default=0.0,
                    help="std of additive Gaussian noise on x and dx independently (absolute scale)")
    ap.add_argument("--noise_seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.noise_seed)

    for role in ("train", "val", "test"):
        files = sorted(glob.glob(f"{args.raw_dir}/{role}_seed*.npy"))
        if not files:
            print(f"[{role}] no files matching {args.raw_dir}/{role}_seed*.npy -- skipping")
            continue
        xs, dxs = [], []
        for f in files:
            d = np.load(f, allow_pickle=True).item()
            xs.append(d["x"])
            dxs.append(d["dx"])
        X = np.concatenate(xs, 0).astype(np.float32)
        dX = np.concatenate(dxs, 0).astype(np.float32)

        # per-probe mean pressure is O(0.1-0.5) (a near-constant offset, not
        # dynamics) and dominates raw RMS -- report fluctuation std (X minus
        # each probe's own mean) instead, since that's the actual signal
        # amplitude the shedding dynamics live in and what noise competes with
        fluct_std = float(X.std(axis=(0, 1)).mean())
        if args.noise_strength:
            X = X + args.noise_strength * rng.randn(*X.shape).astype(np.float32)
            dX = dX + args.noise_strength * rng.randn(*dX.shape).astype(np.float32)

        np.save(f"{args.out_dir}/{role}.npy", {"x": X, "dx": dX})
        print(f"[{role}] {len(files)} trajectories -> x {X.shape}  dx {dX.shape}  "
              f"(fluctuation std {fluct_std:.4f}, noise_strength {args.noise_strength:.4f}"
              + (f", noise/fluctuation {args.noise_strength / fluct_std:.1%}" if fluct_std > 0 else "")
              + ")")

    print(f"\nsaved to {args.out_dir}/")


if __name__ == "__main__":
    main()
