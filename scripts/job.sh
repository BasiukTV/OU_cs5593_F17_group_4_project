#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --mem=8192
#SBATCH --output=preprocess_%J_stdout.log
#SBATCH --error=preprocess_%J_stderr.log
#SBATCH --time=48:00:00
#SBATCH --job-name=github-processing
#SBATCH --mail-user=timo.kaufmann@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --workdir=/scratch/timo
#
#################################################
scratch="/scratch/timo"
indir="${scratch}/github"
outdir="${scratch}/out-${SLURM_JOB_ID}"
mkdir -p "${outdir}"
module load Python/3.5.1-intel-2016a
python3 /home/timo/repo/src/preproc/parse_json.py \
	--indir "${indir}" \
	--outdir "${outdir}" \
	--threads ${SLURM_NTASKS}
