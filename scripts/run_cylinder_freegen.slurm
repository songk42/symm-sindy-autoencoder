#!/bin/bash
#SBATCH --job-name=symm_sindy_cylinder_freegen
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

# cylinder_v5: freeze_generators=0 retry. The original freeze_generators=0
# attempt collapsed the encoder outright (full-rank fraction 0%). cylinder_v2
# through v4 found that lambda_jacobian=0.10-0.15 (jac_warmup_epochs=50) keeps
# the encoder full-rank and well-conditioned with the generator FROZEN at the
# canonical -I. That fixed the collapse, but the free generator converged to
# a 90deg rotation with ||T^p - I|| = 2.83 -- NOT an order-2 element at all.
# Root cause: lambda_order and lambda_repel were both 0 (repo-wide default,
# never overridden anywhere) -- nothing was enforcing T^2=I or repelling the
# search from the trivial T=I attractor (see SINDyAE.py's own comment on this
# exact failure mode). So neither this nor the original freeze_generators=0
# attempt was ever a valid test of generator search. cylinder_v6 turns both
# on for a real test. cylinder_v6 (lambda_repel=0.01) moved the generator
# from a 90deg rotation toward a ~19.5deg rotation (||T-I||=0.479) but never
# cleared the barrier to -I -- ||T^2 - I|| = 0.944, still not order-2. Exactly
# the "spurious attractor at T=I" failure the model code warns about: repel
# was too weak to push it past the identity attractor. cylinder_v7 raises
# lambda_repel 0.01 -> 0.2 to try to clear that barrier.
python3 main.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 nonlinearity=tanh spectral_norm=1 \
    batch_size=16 \
    lambda_dx=0.01 lambda_dz=0.5 lambda_gauge=1.0 gauge_warmup_epochs=20 sequential_threshold=0.001 hidden_dim=128 \
    n_generators=1 symmetry_order=2 freeze_generators=0 lambda_equiv=0.1 \
    lambda_order=0.1 lambda_repel=0.2 equiv_warmup_epochs=100 \
    lambda_jacobian=0.10 jac_warmup_epochs=50 \
    epochs=800 session_name=cylinder_v7 device=0

python3 coordinate_diagnostic.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 \
    nonlinearity=tanh spectral_norm=1 hidden_dim=128 n_generators=1 \
    session_name=cylinder_v7 device=0

python3 phase_d_eval.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 nonlinearity=tanh spectral_norm=1 \
    hidden_dim=128 n_generators=1 symmetry_order=2 freeze_generators=0 \
    session_name=cylinder_v7 device=0
