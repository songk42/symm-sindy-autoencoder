#!/bin/bash
# Expand the cylinder dataset from 2/1/1 to 10/2/2 independent trajectories
# (train/val/test) to give the separating test enough independent data to
# discriminate the true mirror symmetry from decoys (2 train trajectories
# wasn't enough -- see cylinder_v2 result). Split across 2 sequential jobs to
# stay under the 4h partition limit (~40min/trajectory, must run sequentially:
# the apptainer overlay is single-writer). This is job 1 of 2: 4 more train
# (seeds 46-49) + 1 more val (seed 50).
#SBATCH --job-name=cylinder_data_more1
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

$HG python3 create_cylinder.py $COMMON --role train --seed 46
$HG python3 create_cylinder.py $COMMON --role train --seed 47
$HG python3 create_cylinder.py $COMMON --role train --seed 48
$HG python3 create_cylinder.py $COMMON --role train --seed 49
$HG python3 create_cylinder.py $COMMON --role val   --seed 50
