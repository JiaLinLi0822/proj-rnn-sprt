#!/usr/bin/env bash
set -euo pipefail

# SPRT Training Job Submission Script
# ===================================
# This script runs multiple SPRT experiments in parallel with different configurations.
# Each job specifies: sampling_cost, urgency_cost, reward, logLR_range, max_samples, max_steps, num_episodes
#
# Usage:
#   ./submit.sh
#
# Results will be saved in: results/exp_<jobid>/

# ------------------------------------------------------------
# Job configurations: unique_id,sampling_cost,urgency_cost,reward,max_samples,max_steps,num_episodes
# Format: "id,sc,uc,rw,max_samp,max_step,episodes"
# Note: stimuli_logLR is fixed to [-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9]
# 

jobs=(
  "001,0.01,0.00,1.0,10000,10000,1500000"    # infinite horizon
  "002,0.05,0.00,1.0,10000,10000,1500000"    # hard time boundary
  "003,0.01,0.00,1.0,10,10000,1500000"    # soft time boundary
)

# Parallel jobs
MAX_JOBS=3

# Python script
PYTHON_SCRIPT="training.py"

# Results root directory
RESULTS_ROOT="results"

run_one(){
  local PYTHON_SCRIPT="training.py"
  local RESULTS_ROOT="results"
  
  # Parse job parameters: id,sc,uc,rw,max_samp,max_step,episodes
  IFS=',' read -r unique_id sc uc rw max_samp max_step episodes <<< "$1"
  
  # Create descriptive job ID with unique identifier
  jobid="${unique_id}_reward=${rw}_sample_cost=${sc}_urgency_cost=${uc}_logLR=[-0.9,0.9]_max_samples=${max_samp}_max_steps=${max_step}_epNum=${episodes}"
  exp_path="${RESULTS_ROOT}/exp_${jobid}"
  mkdir -p "${exp_path}"

  echo "[START] ${unique_id} - ${jobid}"
  echo "  Unique ID: ${unique_id}"
  echo "  Parameters: sampling_cost=${sc}, urgency_cost=${uc}, reward=${rw}"
  echo "  Max samples=${max_samp}, max_steps=${max_step}, episodes=${episodes}"
  echo "  LogLR fixed: [-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9]"

  conda run -n deeprl python "${PYTHON_SCRIPT}" \
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

  echo "[DONE] ${unique_id} - ${jobid}"
}

export -f run_one

printf "%s\n" "${jobs[@]}" \
  | xargs -P "${MAX_JOBS}" -I {} bash -c 'run_one "$@"' _ {}

echo "All jobs dispatched."