#!/bin/bash
#SBATCH --gres=gpu:v100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=47G
#SBATCH --time=0-02:30:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-cogdrive


echo "starting training"

./prep_data.sh

module load python/3.6
module load scipy-stack
source env_setup.sh

python -u train.py $SLURM_TMPDIR/Saved ./configs/server.ini

#nvidia-smi
