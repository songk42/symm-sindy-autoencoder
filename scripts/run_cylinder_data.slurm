#!/bin/bash
# Generate the cylinder-wake dataset as SEVERAL INDEPENDENT CFD trajectories
# (fix for cylinder_v1 / job 39582330: one short, heavily-overlapping
# trajectory time-sliced into train/val/test gave a held-out EDMD fit so good
# it made the separating-test's relative-cost metric numerically degenerate).
#
# MUST run sequentially, not in parallel: ~/scratch/hydrogym/hg-firedrake
# uses a single apptainer ext3 overlay image, which only one process can hold
# open for writing at a time -- confirmed by testing two concurrent runs
# (second one failed to even mount). Hence one job, one GPU-less CPU node,
# everything in series.
#
#SBATCH --job-name=cylinder_data
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.log

cd /mmfs1/home/songk42/symm-sindy-autoencoder
mkdir -p logs
set -euo pipefail

HG=~/scratch/hydrogym/hg-firedrake
COMMON="--dt 0.02 --tf 100 --t_discard 40 --timesteps 250 --stride 25 --mesh medium --out_dir data/cylinder"
# ~60 post-transient time units/trajectory (~3000 steps) -> ~111 windows each;
# 2 train trajectories + 1 val + 1 test = genuinely independent splits, not
# time-slices of one run. ~33 min/trajectory measured earlier -> ~2.2h total,
# comfortably inside the 4h limit run sequentially.

$HG python3 create_cylinder.py $COMMON --role train --seed 42
$HG python3 create_cylinder.py $COMMON --role train --seed 43
$HG python3 create_cylinder.py $COMMON --role val   --seed 44
$HG python3 create_cylinder.py $COMMON --role test  --seed 45

# Combine into the clean dataset + a noise sweep (cheap: pure numpy on the
# already-simulated trajectories, no extra CFD compute). noise_strength
# values chosen relative to the ~0.02-0.07 per-probe fluctuation std measured
# on the first cylinder run (job 39580546).
source .venv/bin/activate
python3 combine_cylinder.py --out_dir data/cylinder --noise_strength 0.0
for n in 0.01 0.03 0.06; do
    python3 combine_cylinder.py --out_dir "data/cylinder_noise${n}" --noise_strength "$n" --noise_seed 0
done
