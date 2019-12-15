"""
Objective function and its gradients of CEBoost:

L = alpha * L1 + (1 - alpha) * L2
"""
import numpy as np

from ._ci_core import _ci_loss, _ci_grads
from ._efn_core import _efn_loss, _efn_grads

global _ALPHA

def _params_init(params):
    """
    Initializer of global arguments.

    Parameters
    ----------
    params: float
        `alpha` indicates the coefficient in the objective function.
    
    """
    global _ALPHA

    assert params <= 1.0 and params >= .0
    _ALPHA = params

def ce_loss(preds, dtrain):
    """
    Computation of Objective Function.
    L = alpha * L1 + (1 - alpha) * L2.

    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data. This is also known as log hazard ratio.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by: 
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), where N = #data.
    
    Returns
    -------
    tuple:
        Name and value of objective function defined in CEBoost model.

    Notes
    -----
    Absolute value of label represents `T` in survival data, Negtive values are considered 
    right censored, i.e. `E = 0`; Positive values are considered event occurrence, i.e. `E = 1`.
    """
    return "ce_loss", _ALPHA * _efn_loss(preds, dtrain)[1] + (1.0 - _ALPHA) * _ci_loss(preds, dtrain)[1]

def _ce_grads(preds, dtrain):
    """
    Gradient computation of custom objective function.

    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data. This is also known as log hazard ratio.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by: 
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), where N = #data.
    
    Returns
    -------
    tuple:
        The first- and second-order gradients of objective function w.r.t. `preds`.
    """
    L1_grads, L1_hess = _efn_grads(preds, dtrain)
    L2_grads, L2_hess = _ci_grads(preds, dtrain)
    return _ALPHA * L1_grads + (1.0 - _ALPHA) * L2_grads, _ALPHA * L1_hess + (1.0 - _ALPHA) * L2_hess