#!/bin/bash
# Completes job 39582828: that run got train_seed42, train_seed43, val_seed44
# but died on test_seed45 with a transient "disk quota exceeded" (the shared
# stf group quota was momentarily at capacity from other users' jobs, not
# ours -- our own footprint is ~1.1G). Retry just the missing trajectory, then
# run the combine step that never got reached.
#SBATCH --job-name=cylinder_data_finish
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x_%j.log

cd /mmfs1/home/songk42/symm-sindy-autoencoder
mkdir -p logs
set -euo pipefail

HG=~/scratch/hydrogym/hg-firedrake
COMMON="--dt 0.02 --tf 100 --t_discard 40 --timesteps 250 --stride 25 --mesh medium --out_dir data/cylinder"

$HG python3 create_cylinder.py $COMMON --role test --seed 45

source .venv/bin/activate
python3 combine_cylinder.py --out_dir data/cylinder --noise_strength 0.0
for n in 0.01 0.03 0.06; do
    python3 combine_cylinder.py --out_dir "data/cylinder_noise${n}" --noise_strength "$n" --noise_seed 0
done
