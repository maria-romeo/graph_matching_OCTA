#!/bin/bash
 
#SBATCH --job-name=pygm_classic
#SBATCH --output=pygm_classic-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=pygm_classic-%A.err  # Standard error of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --cpus-per-task=16  # Number of CPU cores per task
#SBATCH --time=5-24:00:00


# Set the path to your virtual environment's activate script
VENV_PATH=/u/home/romem/OCTA_time_series/venv_time/bin/activate
# Activate the virtual environment
source $VENV_PATH

# Set your API KEY (wandb) 
API_KEY=0e8c34dc0621c15ee5ad23f91fa243b2d1f12065

# Log in wandb using your API KEY
wandb login $API_KEY

 
# run the program
#/u/home/romem/.conda/envs/OCTA/bin/python3 cnn_octa.py
srun python3 test_pygm_classic.py 
