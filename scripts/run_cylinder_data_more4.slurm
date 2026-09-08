#!/bin/bash
# Job 2 of 2 (see run_cylinder_data_more3.slurm): submitted with
# --dependency=afterany so it only starts once job 1 finishes (the apptainer
# overlay is single-writer -> sequential is required). train seeds 61-65,
# then rebuild the combined clean dataset + noise sweep from ALL trajectories
# now on disk (20 train, 2 val, 2 test).
#SBATCH --job-name=cylinder_data_more4
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

$HG python3 create_cylinder.py $COMMON --role train --seed 61
$HG python3 create_cylinder.py $COMMON --role train --seed 62
$HG python3 create_cylinder.py $COMMON --role train --seed 63
$HG python3 create_cylinder.py $COMMON --role train --seed 64
$HG python3 create_cylinder.py $COMMON --role train --seed 65

source .venv/bin/activate
python3 combine_cylinder.py --out_dir data/cylinder --noise_strength 0.0
for n in 0.01 0.03 0.06; do
    python3 combine_cylinder.py --out_dir "data/cylinder_noise${n}" --noise_strength "$n" --noise_seed 0
done
