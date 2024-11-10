# Interpretable Machine Learning-based Radiation Emulation for ICON

This repository contains the code for the interpretable ML-based Radiation emulator for ICON. The corresponding paper is currently under review in the Journal of Geophysical Research - Machine Learning and Computation 


If you want to use this repository, start by executing
```
conda env create -f environment.yml
conda activate hafner_ml_rad
```

# Repository content
- [evaluation](evaluation) contains some functions for predictiona nd evaluation
- [models](models) contains the NN architecture including preprocessing layer
- [nn_config](nn_config) contains the configuration of all NNs
- [plotter](plotter) contains plotting functions
- [preprocessing](preprocessing) contains the normalization file and data loader
- [trained_nns](trained_nns) contains the trained NNs
- [utils](utils) contains some helper functions
- [submit_levante_gpu.sh](submit_levante_gpu.sh) submits a training and evaluation
- [submit.sh](submit.sh) submits all trainings
- [train_coarse_levante.py](train_coarse_levante.py) contains the training script
- [eval_coare_levante.py](eval_coare_levante.py) contains the evaluation script
- [eval_coare_levante.ipynb](eval_coare_levante.ipynb) contains some further evaluation