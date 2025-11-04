# Interpretable Machine Learning-based Radiation Emulation for ICON


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15276978.svg)](https://doi.org/10.5281/zenodo.15276978)

This repository contains the code for the interpretable ML-based radiation emulator for ICON. The corresponding paper is published:

> Hafner, K., Iglesias-Suarez, F., Shamekh, S., Gentine, P., Giorgetta, M. A., Pincus, R., & Eyring, V. (2025). Interpretable machine learning-based radiation emulation for ICON. *Journal of Geophysical Research: Machine Learning and Computation*, 2, e2024JH000501. [https://doi.org/10.1029/2024JH000501](https://doi.org/10.1029/2024JH000501)


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
