#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=28G

# go to folder and sync venv
cd /home2/nchw73/Year2/Image-Processing/
uv sync
source .venv/bin/activate

# Load W&B credentials from .env
set -a
source .env
set +a

WANDB_ENTITY="theo-farrell99-durham-university"
WANDB_PROJECT="Image-processing"

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is missing in .env"
    exit 1
fi

wandb login "$WANDB_API_KEY"

echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"

# Create wandb sweep and capture the sweep ID from the agent run line
SWEEP_OUTPUT=$(wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" sweep_config.yaml 2>&1)
echo "$SWEEP_OUTPUT"
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "wandb agent" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: could not parse sweep ID from wandb output."
    exit 1
fi

echo "------------------------------------------------------"
echo "Launching agent for sweep: $SWEEP_ID"
echo "------------------------------------------------------"

# Run up to 50 trials; increase --count or remove it to run indefinitely
wandb agent --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" --count 50 "$SWEEP_ID"