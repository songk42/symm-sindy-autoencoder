"""
Phase A coordinate-map diagnostic on a trained checkpoint.

    python3 coordinate_diagnostic.py data_set=synth z_dim=2 u_dim=64 timesteps=200 \
        session_name=<session> model=SINDyAE nonlinearity=tanh spectral_norm=1 device=-1

device=-1 runs on CPU (login node has no GPU), device=mps runs on Apple
Silicon GPU. Reads the checkpoint written by
main.py at trained_models/<data_set>/<model>/<session_name>/checkpoint.pt.
"""

import numpy as np
import torch

from cmd_line import parse_args
from src.utils.other import (get_synth_path, get_lorenz_path,
                             get_lorenz_distort_path, get_cylinder_path,
                             get_checkpoint_path, make_model, get_device)
from src.utils.coordinate_diagnostic import run_diagnostic


CANDIDATES = {
    "synth":          (np.diag([-1.0, -1.0]), 2),
    "lorenz":         (np.diag([-1.0, -1.0, 1.0]), 2),
    "lorenz_distort": (np.diag([-1.0, -1.0, 1.0]), 2),
    # cylinder: no ground-truth z, so z_dim/S are our design choice, not a
    # verified fact -- start with z_dim=2 (two shedding-mode-like dims) and
    # the canonical -I, matching the synth Duffing setup exactly.
    "cylinder":       (np.diag([-1.0, -1.0]), 2),
}

_PATHS = {"synth": get_synth_path, "lorenz": get_lorenz_path,
          "lorenz_distort": get_lorenz_distort_path, "cylinder": get_cylinder_path}


def main():
    args = parse_args()
    device = get_device(args)

    _, paths = _PATHS[args.data_set]()
    data = np.load(paths[1], allow_pickle=True).item()   # val split
    X = np.asarray(data["x"], dtype=np.float32).reshape(-1, np.shape(data["x"])[-1])

    net = make_model(args).to(device)
    cp_path, _ = get_checkpoint_path(args)
    ckpt = torch.load(cp_path, map_location=device)
    net.load_state_dict(ckpt["model"], strict=False)
    print(f"loaded {cp_path} (epoch {ckpt.get('epoch', '?')}), x dim {X.shape[1]}, "
          f"z dim {args.z_dim}, device {device}")

    if args.data_set not in CANDIDATES:
        raise SystemExit(f"no candidate action defined for -DT {args.data_set}")
    S, order = CANDIDATES[args.data_set]
    run_diagnostic(net, X, timesteps=args.timesteps, S=S, order=order)


if __name__ == "__main__":
    main()
