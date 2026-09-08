#!/bin/bash
# Expand the cylinder train set further: 10 -> 20 independent trajectories.
# cylinder_v3 (10 train) still could not get the separating test to
# discriminate the true mirror symmetry from decoys cleanly, so push the
# independent-trajectory count up again. val/test stay at 2 each (held-out
# eval only). Split across 2 sequential jobs (~40min/trajectory, apptainer
# overlay is single-writer -> must run in series). This is job 1 of 2:
# train seeds 56-60.
#SBATCH --job-name=cylinder_data_more3
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:45:00
#SBATCH --output=logs/%x_%j.log

cd /mmfs1/home/songk42/symm-sindy-autoencoder
mkdir -p logs
set -euo pipefail

HG=~/scratch/hydrogym/hg-firedrake
COMMON="--dt 0.02 --tf 100 --t_discard 40 --timesteps 250 --stride 25 --mesh medium --out_dir data/cylinder"

$HG python3 create_cylinder.py $COMMON --role train --seed 56
$HG python3 create_cylinder.py $COMMON --role train --seed 57
$HG python3 create_cylinder.py $COMMON --role train --seed 58
$HG python3 create_cylinder.py $COMMON --role train --seed 59
$HG python3 create_cylinder.py $COMMON --role train --seed 60
