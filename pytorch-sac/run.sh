#!/bin/bash
#SBATCH --account=rrg-dpmeger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem=8G               # memory (per node)
#SBATCH --time=0-23:59            # time (DD-HH:MM)
#SBATCH --array=0-0
#SBATCH --mail-user=lucas.berry@mail.mcgill.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=sac
#SBATCH --output=%x-%j.out


source /home/nwaftp23/pytorch-soft-actor-critic/sac_env/bin/activate

python -u main.py \
	--env Hopper-v2 \
	--seed $SLURM_ARRAY_TASK_ID \
	--modes 0 \
	--automatic_entropy_tuning True
