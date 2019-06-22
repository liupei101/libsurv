from pandas import DataFrame
import numpy as np

def _check_config(config):
    """
    Check configuration and complete it with default_config.

    Parameters
    ----------
    config: dict
        Some configurations or hyper-parameters of neural network.
    """
    default_config = {
        "learning_rate": 0.001,
        "learning_rate_decay": 1.0,
        "activation": "tanh",
        "L2_reg": 0.0,
        "L1_reg": 0.0,
        "optimizer": "sgd",
        "dropout_keep_prob": 1.0,
        "seed": 42
    }
    for k in default_config.keys():
        if k not in config:
            config[k] = default_config[k]

def _check_surv_data(surv_data_X, surv_data_y):
    """
    Check survival data and raise errors.

    Parameters
    ----------
    surv_data_X: DataFrame
        Covariates of survival data.
    surv_data_y: DataFrame
        Labels of survival data. Negtive values are considered right censored.
    """
    if not isinstance(surv_data_X, DataFrame):
        raise TypeError("The type of X must DataFrame.")
    if not isinstance(surv_data_y, DataFrame) or len(surv_data_y.columns) != 1:
        raise TypeError("The type of y must be DataFrame and contains only one column.")

def _prepare_surv_data(surv_data_X, surv_data_y):
    """
    Prepare the survival data. The surv_data will be sorted by abs(`surv_data_y`) DESC.

    Parameters
    ----------
    surv_data_X: DataFrame
        Covariates of survival data.
    surv_data_y: DataFrame
        Labels of survival data. Negtive values are considered right censored. 

    Returns
    -------
    tuple
        sorted indices in `surv_data` and sorted DataFrame of X and y.

    Notes
    -----
    For ensuring the correctness of breslow function computation, survival data
    must be sorted by observed time (DESC).
    """
    _check_surv_data(surv_data_X, surv_data_y)
    # sort by T desc
    T = np.abs(np.squeeze(np.array(surv_data_y)))
    sorted_idx = np.argsort(T)
    return sorted_idx, surv_data_X.iloc[sorted_idx, :], surv_data_y.iloc[sorted_idx, :]