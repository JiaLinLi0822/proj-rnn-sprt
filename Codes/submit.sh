#!/usr/bin/env bash
#SBATCH --job-name=SPRT_train              # Job name
#SBATCH --output=results/exp_%A_%a.out     # Standard output file (%A=job ID, %a=array task ID)
#SBATCH --error=results/exp_%A_%a.err      # Standard error file
#SBATCH --ntasks=1                         # One task per array job
#SBATCH --cpus-per-task=4                  # Request 4 CPU cores per task
#SBATCH --mem=16G                          # Request 16 GB memory per task
#SBATCH --time=12:00:00                    # Time limit: 12 hours
#SBATCH --gres=gpu:1                       # Uncomment this line if a GPU is needed
#SBATCH --array=0-29                       # 3 parameter sets × 10 repeats = 30 array tasks

# Load environment
source ~/.bashrc
conda activate deeprl

# Root directory for results
RESULTS_ROOT="results"

# Three experiment parameter sets:
# sampling_cost, urgency_cost, reward, max_samples, max_steps, num_episodes
jobs=(
  "0.01,0.00,1.0,10000,10000,1500000"    # infinite horizon
  "0.01,0.00,1.0,10,   10,   1500000"    # hard time boundary
  "0.01,0.00,1.0,10,   10000,1500000"    # soft time boundary
)

reps=10
param_idx=$(( SLURM_ARRAY_TASK_ID / reps ))            # yields 0, 1, or 2
rep_idx=$(( SLURM_ARRAY_TASK_ID % reps + 1 ))          # yields 1 through 10

# read parameters
IFS=',' read -r sc uc rw max_samp max_step episodes <<< "${jobs[$param_idx]}"

# jobid and output directory
jobid="reward=${rw}_sample_cost=${sc}_urgency_cost=${uc}_logLR=[-0.9,0.9]_max_samples=${max_samp}_max_steps=${max_step}_epNum=${episodes}_run${rep_idx}"
exp_path="${RESULTS_ROOT}/exp_${jobid}"
mkdir -p "${exp_path}"

echo "[SLURM TASK ${SLURM_ARRAY_TASK_ID}] Parameter set #${param_idx}, run ${rep_idx}"
echo "  sampling_cost=${sc}, urgency_cost=${uc}, reward=${rw}"
echo "  max_samples=${max_samp}, max_steps=${max_step}, num_episodes=${episodes}"

python training.py \
  --jobid "${jobid}" \
  --path "${RESULTS_ROOT}" \
  --hidden_size 64 \
  --num_trials 1 \
  --max_samples "${max_samp}" \
  --max_steps "${max_step}" \
  --num_stimuli 8 \
  --sampling_cost "${sc}" \
  --urgency_cost "${uc}" \
  --reward "${rw}" \
  --num_episodes "${episodes}" \
  --lr 1e-3 \
  --batch_size 256 \
  --gamma 1.0 \
  --lamda 1.0 \
  --beta_v 0.05 \
  --beta_e 0.05 \
  --max_grad_norm 1.0

echo "[SLURM TASK ${SLURM_ARRAY_TASK_ID}] Done."