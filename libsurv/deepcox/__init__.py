import os

from ._model import model

def _safe_mkdir(path):
    """ 
    Create a directory if there isn't one already. 
    """
    try:
        os.mkdir(path)
    except OSError:
        pass

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

def _check_surv_data(surv_data):
    """
    Check survival data and raise errors.

    Parameters
    ----------
    surv_data: dict
        Survival data to be trained in neural network.
    """
    if not isinstance(surv_data, dict):
        raise TypeError("Type of data must be dict.")
    if "X" not in surv_data or "Y" not in surv_data:
        raise KeyError("Data must be a dict and takes 'X' and 'Y' as its keys.")
    if not isinstance(surv_data["Y"], pd.DataFrame) or len(surv_data["Y"].columns) != 1:
        raise ValueError("The label of data must be DataFrame and contains only one column.")

def _surv_data_process(surv_data):
    """
    process the survival data. The surv_data will be sorted by abs(`Y`) DESC.

    Parameters
    ----------
    surv_data: dict
        Survival data to be trained in neural network.

    Returns
    -------
    tuple
        sorted indices in `surv_data` and sorted DataFrame.

    Notes
    -----
    For ensuring the correctness of breslow function computation, survival data
    must be sorted by observed time (DESC).
    """
    _check_surv_data(surv_data)
    # sort by T desc
    T = np.abs(np.array(surv_data["Y"]))
    sorted_idx = np.argsort(T)
    sorted_data = {"X": surv_data["X"].iloc[sorted_idx, :], "Y": surv_data["Y"].iloc[sorted_idx, :]}
    return sorted_idx, sorted_data
