#!/bin/bash
#SBATCH --job-name=symm_sindy
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

# linenc_synth_v2 (job 39537253): recon-pretrain alone did NOT fix it -- R^2
# 0.98/0.91, and the main phase still moved the encoder off the recon-optimal
# chart (recon 1.6e-3 during pretrain -> worse after; on Lorenz base_v2 recon
# went 5e-4 -> 8e-3 the moment dz/gauge turned on). The dz loss reliably
# hijacks the chart and the flexible decoder lets it.
# v3: -RP 300 THEN FREEZE the encoder (-RPF 1, default). The main phase then
# discovers Xi + decoder + (frozen) generator IN the recon-optimal chart
# instead of moving it. Clean test: is a recon-optimal frozen chart good
# enough to read off Duffing?
python3 main.py data_set=synth z_dim=2 u_dim=64 timesteps=200 nonlinearity=tanh \
    linear_encoder=1 recon_pretrain_epochs=300 rp_freeze_encoder=1 \
    lambda_dx=0.01 lambda_dz=0.5 lambda_gauge=1.0 gauge_warmup_epochs=20 \
    sequential_threshold=0.001 hidden_dim=128 \
    n_generators=1 symmetry_order=2 freeze_generators=1 \
    lambda_equiv=0.1 equiv_warmup_epochs=100 \
    epochs=800 session_name=linenc_synth_v3 device=0

python3 coordinate_diagnostic.py data_set=synth z_dim=2 u_dim=64 timesteps=200 \
    nonlinearity=tanh linear_encoder=1 hidden_dim=128 n_generators=1 \
    session_name=linenc_synth_v3 device=0

python3 phase_d_eval.py data_set=synth z_dim=2 u_dim=64 timesteps=200 nonlinearity=tanh \
    linear_encoder=1 hidden_dim=128 n_generators=1 symmetry_order=2 freeze_generators=1 \
    session_name=linenc_synth_v3 baseline_session=dzfit_synth device=0
