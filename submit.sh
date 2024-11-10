#!/bin/sh

#sbatch ./submit_levante_gpu.sh nn_config/SW_HR.yml
#sbatch ./submit_levante_gpu.sh nn_config/LW_HR.yml
sbatch ./submit_levante_gpu.sh nn_config/SW_FLUX.yml
sbatch ./submit_levante_gpu.sh nn_config/LW_FLUX.yml