"""Efron Approximation partial likelihood function and its gradients
"""
import numpy as np
import collections
from functools import cmp_to_key
from lifelines.utils import concordance_index

def _abs_sort(x, y):
    """
    Built-in `cmp` function for sorting.
    """
    x, y = x[1], y[1]
    if abs(x) == abs(y):
        return y - x
    return abs(x) - abs(y)

def _label_abs_sort(label):
    """
    Built-in function for sorting labels according to its absolute value.
    """
    L = [(i, x) for i, x in enumerate(label)]
    L = sorted(L, key=cmp_to_key(_abs_sort))
    return [x[0] for x in L]

def efn_loss(preds, dtrain):
    """
    Computation of objective function.

    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data.

    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by: 
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), where N = #data.

    Returns
    -------
    tuple:
        Name and value of objective function.
    """
    n = preds.shape[0]
    # Sorted Orders
    labels = dtrain.get_label()
    label_order = _label_abs_sort(labels)
    # Statistics for at risk i: R(i)
    R = np.zeros(labels.shape, dtype='int')
    for i, ind in enumerate(label_order):
        if i == 0 or abs(labels[ind]) != abs(labels[label_order[i-1]]):
            R[i] = i
        else:
            R[i] = R[i-1]
    # Compute Loss value
    ### Hazard Ratio
    hr = np.exp(preds)
    ### Compute SR (sum of HR at risk)
    cum_hr = np.cumsum(hr[label_order])
    sum_hr = cum_hr[n-1]
    def SR(idx):
        if R[idx] == 0:
            return sum_hr
        return sum_hr - cum_hr[R[idx]-1]
    out = .0
    cnt_event = 0
    for i, ind in enumerate(label_order):
        if labels[ind] > 0:
            out += np.log(SR(i)) - preds[ind]
            cnt_event += 1
    ### normalize by the number of events
    return "efron_loss", out / cnt_event

def _efn_grads(preds, dtrain):
    """
    Gradient computation of custom objective function - Efron approximation.

    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data.

    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by: 
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), where N = #data.

    Returns
    -------
    tuple:
        The first- and second-order gradients of objective function w.r.t. `preds`.
    """
    n = preds.shape[0]
    # Sorted Orders
    labels = dtrain.get_label()
    label_order = _label_abs_sort(labels)
    # Statistics for at risk i: R(i)
    # Statistics for failures at t: failures(t)
    R = np.zeros(labels.shape, dtype='int')
    failures = collections.OrderedDict()
    for i, ind in enumerate(label_order):
        death_t = labels[ind]
        if i == 0 or abs(death_t) != abs(labels[label_order[i-1]]):
            R[i] = i
        else:
            R[i] = R[i-1]
        if death_t > 0:
            if death_t not in failures:
                failures[death_t] = [i]
            else:
                # ties occured
                failures[death_t].append(i)
    # Compute grad and hessian
    ### Hazard Ratio
    hr = np.exp(preds)
    ### Compute SR (sum of HR at risk)
    cum_hr = np.cumsum(hr[label_order])
    sum_hr = cum_hr[n-1]
    def SR(idx):
        if R[idx] == 0:
            return sum_hr
        return sum_hr - cum_hr[R[idx]-1]
    ### Compute SFR and SFR2
    def SRF(sfr, ord=1):
        sfr[0] = .0
        sfr_prev = sfr[0]
        for t, fails in failures.items():
            sfr[t] = sfr_prev + len(fails) * 1.0 / (SR(fails[0])**ord)
            sfr_prev = sfr[t]
    ### Get SFR and SFR2
    sfr1 = collections.OrderedDict()
    SRF(sfr1, ord=1)
    sfr2 = collections.OrderedDict()
    SRF(sfr2, ord=2)
    ### Compute Gradient and Hessian
    grad = np.zeros_like(preds)
    hess = np.zeros_like(preds)
    death_time = 0
    for ind in label_order:
        death_t = labels[ind]
        if death_t > 0:
            grad[ind] = hr[ind] * sfr1[death_t] - 1.0
            hess[ind] = grad[ind] + 1.0 - (hr[ind]**2) * sfr2[death_t]
            death_time = death_t
        else:
            grad[ind] = hr[ind] * sfr1[death_time]
            hess[ind] = grad[ind] - (hr[ind]**2) * sfr2[death_time]
    ### Finish Compute
    return grad, hess
