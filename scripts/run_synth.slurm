#!/bin/bash
#SBATCH --job-name=symm_sindy_synth
#SBATCH -C rocky8
#SBATCH -p mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x_%j.log

cd /home/songk/symm-sindy-autoencoder
source .venv/bin/activate

module load cuda/13.1.0 || true

set -euo pipefail
export PYTHONUNBUFFERED=1

# synth_jac_v1: pipeline sanity check. The cylinder separating test has
# failed under every lambda_jacobian tried (0.03 collapse, 0.06 near-collapse,
# 0.10/0.15 healthy-but-saturated) -- before reading that as a real physics
# result, confirm the pipeline itself can recover a KNOWN, EXACT symmetry.
# synth = symmetric Duffing oscillator observed through an unknown nonlinear
# map (create_synth.py): z_dim=2 is exact by construction (not a design
# guess like cylinder's), and it carries ground-truth z/dz so Phase D's
# (b)/(c) position/dynamics-closeness checks run for real instead of being
# skipped. Uses the same anti-collapse recipe validated on cylinder
# (lambda_jacobian=0.10, jac_warmup_epochs=50) with freeze_generators=1 to
# test the canonical -I hypothesis honestly first.
#
# u_dim=64, timesteps=200 to match the synth dataset already on disk
# (data/synth/*.npy: train (102400,64) = 512 ICs x 200 steps, val (4000,64)
# = 20 ICs, test (8000,64) = 40 ICs) -- NOT the u_dim=128/timesteps=250 from
# create_synth.py's docstring example, which doesn't match what's generated
# here and crashes the reshape in Datasets.py.
python3 main.py data_set=synth z_dim=2 u_dim=64 timesteps=200 nonlinearity=tanh spectral_norm=1 \
    lambda_dx=0.01 lambda_dz=0.5 lambda_gauge=1.0 gauge_warmup_epochs=20 sequential_threshold=0.001 hidden_dim=128 \
    n_generators=1 symmetry_order=2 freeze_generators=1 lambda_equiv=0.1 equiv_warmup_epochs=100 \
    lambda_jacobian=0.10 jac_warmup_epochs=50 \
    epochs=800 session_name=synth_jac_v1 device=0

python3 coordinate_diagnostic.py data_set=synth z_dim=2 u_dim=64 timesteps=200 \
    nonlinearity=tanh spectral_norm=1 hidden_dim=128 n_generators=1 \
    session_name=synth_jac_v1 device=0

python3 phase_d_eval.py data_set=synth z_dim=2 u_dim=64 timesteps=200 nonlinearity=tanh spectral_norm=1 \
    hidden_dim=128 n_generators=1 symmetry_order=2 freeze_generators=1 \
    session_name=synth_jac_v1 device=0
