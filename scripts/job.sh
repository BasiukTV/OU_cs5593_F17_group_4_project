#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --ntasks=5
#SBATCH --mem=2048
#SBATCH --output=preprocess_%J_stdout.log
#SBATCH --error=preprocess_%J_stderr.log
#SBATCH --time=1:00:00
#SBATCH --job-name=github-processing
#SBATCH --mail-user=timo.kaufmann@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --workdir=/home/timo
#
#################################################
python /home/timo/repo/src/preproc/preprocess.py --directory /scratch/timo/github/
