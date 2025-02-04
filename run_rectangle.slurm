#!/bin/bash -l
#SBATCH --job-name=rectangle_boundaries
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="donisvela@lorentz.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=50G
#SBATCH --time=0-04:00:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-39
 
module load Miniconda3
conda activate general

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD
# Run the file
echo "[$SHELL] ## Run script"
srun python3 rectangle.py $SLURM_ARRAY_TASK_ID
echo "[$SHELL] ## Script finished"
echo "[$SHELL] #### Finished Python test. Have a nice day"
