#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem=8192
#SBATCH --output=out-%J.log
#SBATCH --error=err-%J.log
#SBATCH --time=48:00:00
#SBATCH --job-name=github-processing
#SBATCH --mail-user=timo.kaufmann@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --workdir=/work/timo
#
#################################################
scratch="/work/timo"
indir="${scratch}/parsed"
# outdir="${scratch}/aggregated-${SLURM_JOB_ID}"
mkdir -p "${outdir}"
module load Python/3.5.1-intel-2016a
python3 /home/timo/repo/src/preproc/aggregate.py \
	--file "${indir}/preprocessed.sqlite3"
