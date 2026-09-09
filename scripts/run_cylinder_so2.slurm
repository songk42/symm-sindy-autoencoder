#!/bin/bash
#SBATCH --job-name=symm_sindy_cylinder_so2
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

# cylinder_v8: test general SO(2) equivariance instead of the specific order-2
# (-I) subgroup element. cylinder_v5/v6/v7 tried to force the free generator
# to close at order 2 (T^2=I); it never converged there (v6: 19.5deg rotation,
# ||T^2-I||=0.944; v7: TBD with stronger repel). A limit-cycle oscillator's
# actual symmetry is continuous phase rotation -- -I is just its order-2
# element. This run drops lambda_order to 0 (no order-2 forcing) and keeps
# lambda_repel>0 (still needed to avoid the trivial T=I attractor, or the
# equivariance loss is satisfied vacuously) to see whether ANY rotation angle
# meaningfully linearizes the dynamics.
#
# IMPORTANT for reading results: phase_d_eval's separating test ALWAYS
# hardcodes -I as "the discovered action" for d=2 (see candidate_actions() in
# src/utils/phase_d.py) -- it does NOT substitute whatever this run's free
# generator actually converges to. So Phase D's "SEPARATING TEST" verdict
# here is uninformative for this experiment. What matters is Phase A's
# "learned generator T[0]" block from coordinate_diagnostic.py: compare its
# symmetry cost against the ~1e-2 noise floor established by the fixed wrong
# candidates in prior runs (flip/swap/rot90/shear all scored 1.1e-2 - 2.0e-2
# in cylinder_v2 through v7) -- if this run's learned-T cost lands an order
# of magnitude or more below that floor, that's real evidence of continuous
# equivariance even without order-2 closure.
python3 main.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 nonlinearity=tanh spectral_norm=1 \
    batch_size=16 \
    lambda_dx=0.01 lambda_dz=0.5 lambda_gauge=1.0 gauge_warmup_epochs=20 sequential_threshold=0.001 hidden_dim=128 \
    n_generators=1 symmetry_order=2 freeze_generators=0 lambda_equiv=0.1 \
    lambda_order=0 lambda_repel=0.2 equiv_warmup_epochs=100 \
    lambda_jacobian=0.10 jac_warmup_epochs=50 \
    epochs=800 session_name=cylinder_v8 device=0

python3 coordinate_diagnostic.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 \
    nonlinearity=tanh spectral_norm=1 hidden_dim=128 n_generators=1 \
    session_name=cylinder_v8 device=0

python3 phase_d_eval.py data_set=cylinder z_dim=2 u_dim=20 timesteps=250 nonlinearity=tanh spectral_norm=1 \
    hidden_dim=128 n_generators=1 symmetry_order=2 freeze_generators=0 \
    session_name=cylinder_v8 device=0
