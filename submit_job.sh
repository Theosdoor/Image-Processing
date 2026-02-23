#!/bin/bash
#SBATCH --job-name=SAEs
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=12:00:00
#SBATCH --mem=28G

# go to folder and sync venv
cd /home2/nchw73/Year4/L4_Project/list-comp-priv/
uv sync
source .venv/bin/activate

# verify we got gpu
echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"


# Run the experiments
python3 train_btk_sae.py