# libsurv

## Introduction

A library of efficient survival analysis models, including `DeepCox`, `HitBoost`, `CEBoost` and `EfnBoost` methods.

- `DeepCox`: Deep cox proportional hazard model implemented by tensorflow. It's exactly the same as [`TFDeepSurv`](https://github.com/liupei101/TFDeepSurv).
- `HitBoost`: Survival analysis via a multi-output gradient boosting decision tree method.
- `EfnBoost`: Optimized cox proportional hazard model via an objective function of Efron approximation.
- `CEBoost`: Adding convex function approximated concordance index in `EfnBoost` to adjust risk ranking.

## Enhancement

- comprehensive document
- python package distribution

## Installation

```bash
# in the directory where `setup.py` is located
ls
# install via pip or pip3 (only support for python>=3.5)
pip3 install .
```

## Usage

Usage of `DeepCox`, `EfnBoost`, `CEBoost` and `HitBoost` are provided in [Jupyter Notebooks](examples/).

## Citation

If you would like to cite our package, some reference papers are listed below: 
- HitBoost(*Accepted*): [HitBoost: Survival Analysis via a Multi-output Gradient Boosting Decision Tree method.](https://doi.org/10.1109/ACCESS.2019.2913428)
- EfnBoost(*Under Review*): Optimizing Survival Analysis of XGBoost for Ties to Predict Prognostic Status of Breast Cancer
- DeepCox(*Under Review*): Deep Survival Learning for Predicting the Overall Survival in Breast Cancer using Clinical and Follow-up Data
