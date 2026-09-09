#!/bin/bash
#SBATCH --job-name=symm_sindy_cylinder
#SBATCH -C rocky8
#SBATCH -p mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.log

cd /home/songk/symm-sindy-autoencoder
source .venv/bin/activate

module load cuda/13.1.0 || true

set -euo pipefail
export PYTHONUNBUFFERED=1

# cylinder_v2/v3: multi-trajectory dataset (train = 2 independent CFD
# trajectories, seeds 42/43, 2220 windows; val = 1 trajectory, seed 44, 222
# windows; test = 1 trajectory, seed 45, 222 windows). The real independent
# seed-45 test trajectory has landed and is confirmed genuinely distinct from
# val (different checksum/content, verified 2026-09-08) -- no longer a
# val.npy copy stopgap. This fixes cylinder_v1's problem (one
# short correlated trajectory time-sliced into train/val/test made the
# separating-test's held-out fit implausibly good, degenerating the metric) --
# train and val are now genuinely independent trajectories.
# No ground-truth latent here (see src/utils/phase_d.py has_truth gating) --
# z_dim=2 and S=-I are a DESIGN CHOICE (two shedding-mode-like dims, canonical
# involution), not a verified fact. batch_size lowered from the default 50
# given the still-modest window count. Recipe = the best synth config
# (phaseC_jac_v1).
#
# cylinder_v2 (lambda_jacobian=0.15, jac_warmup_epochs=50) fixed the collapse
# from an earlier freeze_generators=0 attempt (encoder now full-rank,
# condition number ~1.0), but overcorrected: the encoder got so rigid/
# near-isometric that the separating test saturated -- discovered (-I) cost
# 1.87e-2 vs shear 1.69e-2, all within noise of the 0.02 threshold, so the
# test can no longer tell -I apart from arbitrary linear maps. It also
# starved the SINDy dynamics fit (dz MSE plateaued ~7-9e-3 the whole run,
# 30-80x worse than earlier runs). Lowered lambda_jacobian to 0.06, which
# undershot the other way: near-collapse again (min singular value 1.7e-4,
# condition number 5.7e3), separating test degenerate via the collapse
# mechanism instead. Bisecting to 0.10 between the two failure modes.
python3 main.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 nonlinearity=tanh spectral_norm=1 \
    batch_size=16 \
    lambda_dx=0.01 lambda_dz=0.5 lambda_gauge=1.0 gauge_warmup_epochs=20 sequential_threshold=0.001 hidden_dim=128 \
    n_generators=1 symmetry_order=2 freeze_generators=1 lambda_equiv=0.1 equiv_warmup_epochs=100 \
    lambda_jacobian=0.10 jac_warmup_epochs=50 \
    epochs=800 session_name=cylinder_v4 device=0

python3 coordinate_diagnostic.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 \
    nonlinearity=tanh spectral_norm=1 hidden_dim=128 n_generators=1 \
    session_name=cylinder_v4 device=0

python3 phase_d_eval.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 nonlinearity=tanh spectral_norm=1 \
    hidden_dim=128 n_generators=1 symmetry_order=2 freeze_generators=1 \
    session_name=cylinder_v4 device=0
