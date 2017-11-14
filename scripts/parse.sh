#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --cpus-per-task=15
#SBATCH --mem=4096
#SBATCH --output=out-%J.log
#SBATCH --error=err-%J.log
#SBATCH --time=01:00:00
#SBATCH --job-name=github-processing
#SBATCH --mail-user=timo.kaufmann@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --workdir=/work/timo
#
#################################################
scratch="/work/timo"
indir="${scratch}/github"
outdir="${scratch}/parsed-${SLURM_JOB_ID}"
mkdir -p "${outdir}"
module load Python/3.5.1-intel-2016a
python3 /home/timo/repo/src/preproc/parse_json.py \
	--indir "${indir}" \
	--outdir "${outdir}" \
	--threads "${SLURM_CPUS_ON_NODE}"
