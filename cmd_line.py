import sys
from argparse import Namespace
from pathlib import Path

import hydra
from omegaconf import OmegaConf

_CONFIG_DIR = str(Path(__file__).resolve().parent / "conf")


def parse_args():
    """Load run configuration via Hydra (see conf/config.yaml) and return it as
    an argparse.Namespace, so every existing call site (main.py, create_synth.py,
    phase_d_eval.py, coordinate_diagnostic.py, koopman_diagnostic.py, ...) keeps
    working unchanged: args.z_dim, getattr(args, "linear_encoder", 0), vars(args),
    args.__dict__, setattr(args, ...) all behave exactly as before.

    CLI overrides use Hydra's key=value syntax against the flat keys in
    conf/config.yaml, e.g.:
        python3 main.py data_set=synth z_dim=2 lambda_dx=0.01 session_name=my_run
    The old argparse flags (-DT, -Z, -L1, -sess, ...) no longer work.

    Uses the Compose API (hydra.initialize_config_dir + hydra.compose) rather
    than the @hydra.main decorator, so there is no Hydra-managed output
    directory / working-directory change -- relative paths (./experiments/,
    ./trained_models/, ...) behave exactly as they did under argparse.
    """
    with hydra.initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = hydra.compose(config_name="config", overrides=sys.argv[1:])
    return Namespace(**OmegaConf.to_container(cfg, resolve=True))
