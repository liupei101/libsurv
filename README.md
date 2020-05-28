# libsurv

## Introduction

A library of efficient survival analysis models, including `DeepCox`, `HitBoost`, `BecCox` and `EfnBoost` methods.

- `DeepCox`: Deep cox proportional hazard model implemented by tensorflow. It's exactly the same as [`TFDeepSurv`](https://github.com/liupei101/TFDeepSurv).
- `HitBoost`: Survival analysis via a multi-output gradient boosting decision tree method.
- `EfnBoost`: Optimized cox proportional hazard model via an objective function of Efron approximation.
- `BecCox`: Adding convex function approximated concordance index in `EfnBoost` to adjust risk ranking.

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

Usage of `DeepCox`, `EfnBoost`, `BecCox` and `HitBoost` are provided in [Jupyter Notebooks](examples/).

Hyper-parameters tuning can refer to [libsurv/bysopt](bysopt/).

## Citation

If you would like to cite our package, some reference papers are listed below: 
- HitBoost(*Accepted by IEEE-Access*): [P. Liu, B. Fu and S. X. Yang, "HitBoost: Survival Analysis via a Multi-Output Gradient Boosting Decision Tree Method," in IEEE Access, vol. 7, pp. 56785-56795, 2019, doi: 10.1109/ACCESS.2019.2913428.](https://doi.org/10.1109/ACCESS.2019.2913428)
- EfnBoost(*Accepted by IEEE-TBME*): [P. Liu, B. Fu, S. X. Yang, L. Deng, X. Zhong and H. Zheng, "Optimizing Survival Analysis of XGBoost for Ties to Predict Disease Progression of Breast Cancer," in IEEE Transactions on Biomedical Engineering, doi: 10.1109/TBME.2020.2993278.](https://doi.org/10.1109/TBME.2020.2993278)
- DeepCox(*Under Review*): Deep Survival Learning for Predicting the Overall Survival in Breast Cancer using Clinical and Follow-up Data
