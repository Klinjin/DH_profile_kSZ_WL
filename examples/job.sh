#!/bin/bash
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -A m4031
#SBATCH -J physics_nn_training
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=lindazjin@berkeley.edu
#SBATCH --mem=120G

# CPU-only environment settings
export OMP_NUM_THREADS=32
export OMP_PLACES=threads  
export OMP_PROC_BIND=spread
export JAX_PLATFORMS=cpu  # Force JAX to use CPU only
export JAX_ENABLE_X64=false  # Use float32 for better performance

# Load environment
module load python
source /global/u1/l/lindajin/virtualenvs/env1/bin/activate

# Change to project directory
cd /pscratch/sd/l/lindajin/DH_profile_kSZ_WL

# Run CPU-optimized physics-informed neural network training
python -u examples/physics_neural_demo.py