"""
Phase D -- held-out separating evaluation + equation recovery for a trained
coordinate-problem checkpoint.

    python3 phase_d_eval.py data_set=synth z_dim=2 u_dim=64 timesteps=200 \
        nonlinearity=tanh spectral_norm=1 hidden_dim=128 \
        n_generators=1 symmetry_order=2 freeze_generators=1 \
        session_name=phaseC_frozen_v3 baseline_session=dzfit_synth device=-1

device=-1 runs on CPU, device=mps runs on Apple Silicon GPU, device=N runs on
CUDA device N. The CLI overrides must describe the ARCHITECTURE of the
session_name checkpoint (so the model can be rebuilt); baseline_session names
an unconstrained model of the same architecture whose args.txt is read for its
own config. Uses the TEST split, which was untouched by training and by the
Phase A / C diagnostics.
"""

import json
from argparse import Namespace

import numpy as np
import torch

from cmd_line import parse_args
from src.utils.other import (get_synth_path, get_lorenz_path,
                             get_lorenz_distort_path, get_cylinder_path,
                             get_checkpoint_path, make_model, get_device)
from src.utils.phase_d import run_phase_d, true_duffing_coeffs


_PATHS = {"synth": get_synth_path, "lorenz": get_lorenz_path,
          "lorenz_distort": get_lorenz_distort_path, "cylinder": get_cylinder_path}


def _true_coeffs(data_set):
    """Generating-ODE coefficients in poly_features ordering, or None."""
    if data_set == "synth":
        return true_duffing_coeffs(delta=0.1)
    if data_set == "lorenz_distort":
        from src.dataset.lorenz_functions import lorenz_coefficients, _LORENZ_NORM
        return lorenz_coefficients(_LORENZ_NORM, poly_order=3)
    return None


def _load(net, cp_path, device):
    ckpt = torch.load(cp_path, map_location=device)
    net.load_state_dict(ckpt["model"], strict=False)
    return net.to(device).eval(), ckpt.get("epoch", "?")


def _baseline(args, device):
    """Rebuild + load the unconstrained baseline from its own args.txt."""
    base_args = Namespace(**vars(args))
    base_args.session_name = args.baseline_session
    cp_path, folder = get_checkpoint_path(base_args)
    # architecture-affecting flags must come from the baseline's own config, not
    # the -sess model's CLI flags; default them off for pre-existing checkpoints
    for k in ("n_generators", "freeze_generators", "kinematic_row",
              "linear_encoder"):
        setattr(base_args, k, 0)
    try:
        with open(folder + "args.txt") as f:
            saved = json.load(f)
        for k, v in saved.items():
            setattr(base_args, k, v)
    except FileNotFoundError:
        print(f"[baseline] no args.txt at {folder}; assuming an unconstrained model")
    base_args.session_name = args.baseline_session
    net = make_model(base_args)
    net, epoch = _load(net, cp_path, device)
    print(f"loaded baseline {cp_path} (epoch {epoch}, n_generators="
          f"{getattr(base_args, 'n_generators', 0)})")
    return net


def main():
    args = parse_args()
    device = get_device(args)

    _, paths = _PATHS[args.data_set]()
    test_data = np.load(paths[2], allow_pickle=True).item()   # TEST split
    for k in ("x", "dx", "dz", "z"):
        if k in test_data:
            test_data[k] = np.asarray(test_data[k]).reshape(-1, np.shape(test_data[k])[-1])

    net = make_model(args)
    cp_path, _ = get_checkpoint_path(args)
    net, epoch = _load(net, cp_path, device)
    print(f"loaded {cp_path} (epoch {epoch}), x dim {test_data['x'].shape[1]}, "
          f"z dim {args.z_dim}, device {device}")

    baseline_net = _baseline(args, device) if args.baseline_session else None

    run_phase_d(net, test_data, timesteps=args.timesteps,
                degree=args.poly_order, baseline_net=baseline_net,
                true_coeffs=_true_coeffs(args.data_set))


if __name__ == "__main__":
    main()
