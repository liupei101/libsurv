"""
Objective function and its gradients of CEBoost:

L = alpha * L1 + (1 - alpha) * L2
"""
import numpy as np

from ._ci_core import ci_loss, _ci_grads
from ._efn_core import efn_loss, _efn_grads

from ..utils import concordance_index

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
    # NOTE: If params equals 0, the first boosting round in L2 would get errors.
    #       The initial zero prediction in XGBoost results in this issue.
    if params == .0:
        params = 1e-2
    
    _ALPHA = params

def ce_evals(preds, dtrain):
    """
    Evaluation of CEBoost model.

    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data. This is also known as log hazard ratio.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by: 
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), where N = #data.
    
    Returns
    -------
    float:
        Concordance index.
    """
    return "ce_evals", concordance_index(dtrain.get_label(), preds)

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
    __, L1_loss = efn_loss(preds, dtrain)
    __, L2_loss = ci_loss(preds, dtrain)
    return "ce_loss", _ALPHA * L1_loss + (1.0 - _ALPHA) * L2_loss

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