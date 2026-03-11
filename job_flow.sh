#!/bin/bash
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -A m4031
#SBATCH -J Pk_flow_training
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=lindazjin@berkeley.edu
#SBATCH --output=mean_Pk_flow_training_%J.out
#SBATCH --error=mean_Pk_flow_training_%J.err
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

# Define run name (can match job name or be different)
RUN_NAME="mean_Pk_flow_1000_epochs_32_batch_smoother"

python /pscratch/sd/l/lindajin/DH_profile_kSZ_WL/src/script/flow_train.py \
    --max_epochs 1000 \
    --run_name ${RUN_NAME} \
    --batch_size 32 \
    --n_samples 100 \
    --mean_profile 
