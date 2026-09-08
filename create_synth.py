"""
Generate the synthetic 'coordinate problem' dataset (symmetric Duffing oscillator
observed through an unknown nonlinear map). See src/dataset/synth_functions.py.

Usage:
    python3 create_synth.py z_dim=2 u_dim=128 timesteps=250 \
        train_initial_conds=2048 val_initial_conds=20 test_initial_conds=100
"""

import os
import numpy as np
from cmd_line import parse_args
from src.dataset.synth_functions import get_duffing_data
from src.utils.other import get_synth_path


def main():
    args = parse_args()
    dt = 0.02

    if args.z_dim != 2:
        raise SystemExit(f"synth is a 2D Duffing system; pass z_dim=2 (got z_dim={args.z_dim})")

    common = dict(timesteps=args.timesteps, u_dim=args.u_dim, dt=dt,
                  noise_strength=args.noise_strength, psi_seed=0,
                  psi_scale=args.psi_scale)

    train_data = get_duffing_data(n_ics=args.train_initial_conds, ic_seed=0, **common)
    val_data = get_duffing_data(n_ics=args.val_initial_conds, ic_seed=1, **common)
    test_data = get_duffing_data(n_ics=args.test_initial_conds, ic_seed=2, **common)

    folder, data_paths = get_synth_path()
    if not os.path.isdir(folder):
        os.system("mkdir -p " + folder)
    np.save(data_paths[0], train_data)
    np.save(data_paths[1], val_data)
    np.save(data_paths[2], test_data)
    print("saved synth dataset to", folder, f"(psi_scale={args.psi_scale})")
    print("  x:", train_data['x'].shape, " dz:", train_data['dz'].shape)
    _report_psi_linearity(test_data)


def _report_psi_linearity(data):
    """Upper bound on an affine encoder: R^2 of a plain linear x -> z_true fit.
    ~1 => psi^-1 is linear (easy); << 1 => psi is genuinely nonlinear (hard)."""
    x = np.asarray(data['x'], float); z = np.asarray(data['z'], float)
    A = np.hstack([x, np.ones((len(x), 1))])
    M, *_ = np.linalg.lstsq(A, z, rcond=None)
    r2 = 1 - ((z - A @ M) ** 2).sum(0) / ((z - z.mean(0)) ** 2).sum(0)
    print("  linear x->z_true R^2 (psi^-1 linearity):", np.round(r2, 5))


if __name__ == '__main__':
    main()
