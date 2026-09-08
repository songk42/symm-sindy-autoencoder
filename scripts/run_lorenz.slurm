#!/bin/bash
#SBATCH --job-name=symm_sindy_lorenz
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x_%j.log

cd /mmfs1/home/songk42/symm-sindy-autoencoder
source .venv/bin/activate

module load cuda/12.9.1 || true

set -euo pipefail
export PYTHONUNBUFFERED=1

# NOTE: this was "-NG 0 ... -NG 1 ..." (a leftover duplicate flag from -sess
# lorenz_dzfit_v2's base config) -- argparse silently took the last -NG (1);
# Hydra rejects duplicate keys outright, so the dedup below preserves the
# actual effective config (n_generators=1) exactly.
python3 main.py data_set=lorenz_distort z_dim=3 u_dim=64 timesteps=250 nonlinearity=tanh hidden_dim=128 \
    lambda_dx=0.001 lambda_dz=0.05 lambda_gauge=3.0 gauge_warmup_epochs=50 \
    n_generators=1 symmetry_order=2 freeze_generators=1 \
    lambda_equiv=0.1 equiv_warmup_epochs=100 \
    lambda_jacobian=0.03 jac_warmup_epochs=150 \
    epochs=800 session_name=lorenz_dzfit_v2 device=0
