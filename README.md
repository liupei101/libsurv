# libsurv

## Introduction

A library of efficient survival analysis models, including DeepCox, HitBoost and EfnBoost methods.

- `DeepCox`: Deep cox proportional hazard model implemented by tensorflow.
- `HitBoost`: Survival analysis via a multi-output gradient boosting decision tree method.
- `EfnBoost`: Optimized cox proportional hazard model via an objective function of Efron approximation.

## Installation

```bash
# in the directory where `setup.py` is located
ls
# install via pip or pip3 (only support for python>=3.5)
pip3 install .
```

## Usage

Usage of `DeepCox`, `EfnBoost` and `HitBoost` are provided in [Jupyter Notebooks](examples/).

## Citation

If you would like to cite our package, some reference papers are listed below: 
- HitBoost(*Accepted*): [HitBoost: Survival Analysis via a Multi-output Gradient Boosting Decision Tree method.](https://doi.org/10.1109/ACCESS.2019.2913428)
- EfnBoost(*Under Review*)
- DeepCox(*Under Review*)
