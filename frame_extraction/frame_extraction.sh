#!/bin/bash
#SBATCH --cpus-per-task=4 -n1 --mem-per-cpu=8G
module load anaconda3
python3 frame_extraction.py $SLURM_ARRAY_TASK_ID
