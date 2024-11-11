# Interpretable Machine Learning-based Radiation Emulation for ICON

This repository contains the code for the interpretable ML-based radiation emulator for ICON. The corresponding paper is currently under review in the Journal of Geophysical Research - Machine Learning and Computation 


If you want to use this repository, start by executing
```
conda env create -f environment.yml
conda activate hafner_ml_rad
```

# Repository content
- [evaluation](evaluation) contains some functions for prediction and evaluation
- [models](models) contains the NN architecture including preprocessing layer
- [nn_config](nn_config) contains the configuration of all NNs
- [plotter](plotter) contains plotting functions
- [preprocessing](preprocessing) contains the normalization file and data loader
- [utils](utils) contains some helper functions
- [train_coarse_levante.py](train_coarse_levante.py) contains the training script
- [eval_coarse_levante.py](eval_coarse_levante.py) contains the evaluation script
- [eval_coarse_levante.ipynb](eval_coarse_levante.ipynb) contains some further evaluation