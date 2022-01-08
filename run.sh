#!/bin/bash
#SBATCH --account=pcon0023
#SBATCH --job-name=LP
#SBATCH --time=10:00:00
#SBATCH --nodes=1 --ntasks-per-node=40
#SBATCH --output=run.out

echo "Job started..."
python run.py --eval_metrics all 
echo "Job finished"