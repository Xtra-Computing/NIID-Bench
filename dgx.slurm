#!/bin/bash

#SBATCH --job-name=small_test
#SBATCH --partition=small
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=diaoyiqun@sjtu.edu.cn
#SBATCH --output=output.txt
#SBATCH --error=err.txt

module load gcc cuda

sh run.sh
