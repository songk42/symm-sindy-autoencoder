"""
Generate the distorted-Lorenz 'coordinate problem' dataset: the Lorenz-63 latent
observed through an unknown random tanh MLP (so the Z2 symmetry S = diag(-1,-1,1)
is linear in z but nonlinear in x). See src/dataset/lorenz_functions.py.

Usage:
    python3 create_lorenz_distort.py z_dim=3 u_dim=64 timesteps=250 \
        train_initial_conds=2048 val_initial_conds=20 test_initial_conds=100
"""

import os
import numpy as np
from cmd_line import parse_args
from src.dataset.lorenz_functions import get_lorenz_distorted
from src.utils.other import get_lorenz_distort_path


def main():
    args = parse_args()
    dt = 0.02

    if args.z_dim != 3:
        raise SystemExit(f"distorted Lorenz is a 3D system; pass z_dim=3 (got z_dim={args.z_dim})")

    common = dict(timesteps=args.timesteps, u_dim=args.u_dim, dt=dt,
                  noise_strength=args.noise_strength, psi_seed=0,
                  psi_scale=args.psi_scale)

    train_data = get_lorenz_distorted(n_ics=args.train_initial_conds, ic_seed=0, **common)
    val_data = get_lorenz_distorted(n_ics=args.val_initial_conds, ic_seed=1, **common)
    test_data = get_lorenz_distorted(n_ics=args.test_initial_conds, ic_seed=2, **common)

    folder, data_paths = get_lorenz_distort_path()
    if not os.path.isdir(folder):
        os.system("mkdir -p " + folder)
    np.save(data_paths[0], train_data)
    np.save(data_paths[1], val_data)
    np.save(data_paths[2], test_data)
    print("saved distorted-Lorenz dataset to", folder, f"(psi_scale={args.psi_scale})")
    print("  x:", train_data['x'].shape, " dz:", train_data['dz'].shape,
          " z range:", train_data['z'].min(), train_data['z'].max())
    x = np.asarray(test_data['x'], float); z = np.asarray(test_data['z'], float)
    A = np.hstack([x, np.ones((len(x), 1))])
    M, *_ = np.linalg.lstsq(A, z, rcond=None)
    r2 = 1 - ((z - A @ M) ** 2).sum(0) / ((z - z.mean(0)) ** 2).sum(0)
    print("  linear x->z_true R^2 (psi^-1 linearity):", np.round(r2, 5))


if __name__ == '__main__':
    main()
