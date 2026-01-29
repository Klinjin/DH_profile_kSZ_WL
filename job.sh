#!/bin/bash
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -A m4031
#SBATCH -J Pk_nn_training
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=lindazjin@berkeley.edu
#SBATCH --output=flow_nn_training.out
#SBATCH --error=flow_nn_training.err
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


python /pscratch/sd/l/lindajin/DH_profile_kSZ_WL/src/flow_train.py
# python /pscratch/sd/l/lindajin/DH_profile_kSZ_WL/src/models/pk_neural_emulator.py

# # Run CPU-optimized physics-informed neural network training
#   python example/physics_nn_training.py \
#     --data_dir data/simulation_data/ \
#     --output_dir trained_models/PhysicsNNTrainer_$(date +%m%d%H_%M)_CAP_gas_full_bins/ \
#     --n_epochs 5000 \
#     --batch_size 64 \
#     --learning_rate 0.001 \
#     --hidden_layers 3 \
#     --hidden_units 128 \
#     --dropout_rate 0.1 \
#     --weight_decay 1e-5