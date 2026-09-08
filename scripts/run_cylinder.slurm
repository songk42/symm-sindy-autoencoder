#!/bin/bash
#SBATCH --job-name=symm_sindy_cylinder
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

# cylinder_v2: PRELIMINARY look at the multi-trajectory dataset (train = 2
# independent CFD trajectories, seeds 42/43, 222 windows; val = 1 trajectory,
# seed 44, 111 windows). test.npy is currently val.npy COPIED as a stopgap --
# the real independent test trajectory (seed 45) is still being generated in
# a background job; re-run phase_d_eval once it lands (data/cylinder/test.npy
# will be regenerated for real then). This fixes cylinder_v1's problem (one
# short correlated trajectory time-sliced into train/val/test made the
# separating-test's held-out fit implausibly good, degenerating the metric) --
# train and val are now genuinely independent trajectories.
# No ground-truth latent here (see src/utils/phase_d.py has_truth gating) --
# z_dim=2 and S=-I are a DESIGN CHOICE (two shedding-mode-like dims, canonical
# involution), not a verified fact. batch_size lowered from the default 50
# given the still-modest window count. Recipe = the best synth config
# (phaseC_jac_v1).
python3 main.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 nonlinearity=tanh spectral_norm=1 \
    batch_size=16 \
    lambda_dx=0.01 lambda_dz=0.5 lambda_gauge=1.0 gauge_warmup_epochs=20 sequential_threshold=0.001 hidden_dim=128 \
    n_generators=1 symmetry_order=2 freeze_generators=1 lambda_equiv=0.1 equiv_warmup_epochs=100 \
    lambda_jacobian=0.03 jac_warmup_epochs=150 \
    epochs=800 session_name=cylinder_v2 device=0

python3 coordinate_diagnostic.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 \
    nonlinearity=tanh spectral_norm=1 hidden_dim=128 n_generators=1 \
    session_name=cylinder_v2 device=0

python3 phase_d_eval.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 nonlinearity=tanh spectral_norm=1 \
    hidden_dim=128 n_generators=1 symmetry_order=2 freeze_generators=1 \
    session_name=cylinder_v2 device=0
