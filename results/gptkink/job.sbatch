#!/bin/bash
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --account=paquettee-2025
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --propagate=NONE
#SBATCH -t 0-23:50
#SBATCH -o results/gptkink/slurm/%x.%j.out
#SBATCH -e results/gptkink/slurm/%x.%j.err
#SBATCH --signal=B:USR1@300
# Generic job for the gptkink experiment: runs the command given as arguments from the
# repository root. Submit from the repository root (the log paths above are relative):
#     sbatch --job-name NAME results/gptkink/job.sbatch lm/bin/python experiments/gptkink_train.py ...
# Five minutes before the time limit slurm sends USR1; it is forwarded to the python
# process, which checkpoints and exits. Re-submitting the same command resumes the run.
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "job $SLURM_JOB_ID on $(hostname) at $(date): $*"
"$@" &
PID=$!
trap 'kill -USR1 $PID; wait $PID' USR1 TERM
wait $PID
echo "exit $? at $(date)"
