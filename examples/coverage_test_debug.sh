#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00  # Increased to 2 hours
#SBATCH -A m4031
#SBATCH -J gp_coverage_debug
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=lindazjin@berkeley.edu
#SBATCH --output=gp_coverage_debug_%j.out
#SBATCH --error=gp_coverage_debug_%j.err
#SBATCH --mem=60G  # Reduced memory to avoid allocation issues

# Print system info
echo "=== JOB STARTED ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working dir: $(pwd)"

# Load environment
echo "Loading environment..."
module load python
source /global/u1/l/lindajin/virtualenvs/env1/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Change to project directory
cd /pscratch/sd/l/lindajin/DH_profile_kSZ_WL
echo "Changed to: $(pwd)"

# Set JAX environment
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=False
export CUDA_VISIBLE_DEVICES=''

# Test basic functionality first
echo "=== TESTING BASIC FUNCTIONALITY ==="
python -c "
import sys
sys.path.append('/pscratch/sd/l/lindajin/DH_profile_kSZ_WL')
from src.inference.gp_coverage_test import main
print('✅ Basic import works')
"

if [ $? -ne 0 ]; then
    echo "❌ Basic functionality test failed"
    exit 1
fi

# Run with minimal parameters first
echo "=== RUNNING MINIMAL TEST ==="
python src/inference/gp_coverage_test.py \
    --gp_dir /pscratch/sd/l/lindajin/DH_profile_kSZ_WL/trained_gp_models/GPTrainer_092225_1900_CAP_gas \
    --n_sims 2 \
    --n_samples 50 \
    --burnin 10 \
    --n_params 6

if [ $? -eq 0 ]; then
    echo "✅ Minimal test successful"

    # If minimal test works, try larger test
    echo "=== RUNNING FULL TEST ==="
    python src/inference/gp_coverage_test.py \
        --gp_dir /pscratch/sd/l/lindajin/DH_profile_kSZ_WL/trained_gp_models/GPTrainer_092225_1900_CAP_gas \
        --n_sims 5 \
        --n_samples 500 \
        --burnin 100 \
        --n_params 6

    echo "Coverage test completed with exit code: $?"
else
    echo "❌ Minimal test failed with exit code: $?"
fi

echo "=== JOB COMPLETED ==="
echo "End time: $(date)"