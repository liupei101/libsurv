"""
L1 term of objective function in BecCox.

Efron Approximation partial likelihood function and its gradients
"""
import numpy as np
from functools import cmp_to_key

def _abs_sort(x, y):
    """
    Built-in `cmp` function for sorting.
    """
    x, y = x[1], y[1]
    if abs(x) == abs(y):
        return x - y
    return abs(x) - abs(y)

def _label_abs_sort(label):
    """
    Sort labels according to its absolute value.

    Put small absolute value at first. If the absolute values equal, then 
    put the negtive one at first.

    Parameters
    ----------
    label: list

    Returns
    -------
    list:
        The index of sorted list.
    """
    L = [(idx, val) for idx, val in enumerate(label)]
    L = sorted(L, key=cmp_to_key(_abs_sort))
    return [item[0] for item in L]

def efn_loss(preds, dtrain):
    """
    Computation of objective function.
    a.k.a Negtive Log of Efron Approximation.
    
    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by: 
        `labels = dtrain.get_label()`. It is `numpy.array` with shape of (N, ), 
        where N = #data. Negtive values are considered right censored (E = 0).

    Returns
    -------
    tuple:
        Name and value of objective function.
    """
    # prediction
    n = preds.shape[0]
    
    # lables
    labels = dtrain.get_label()
    sorted_index = _label_abs_sort(labels)
    E = (labels[sorted_index] > 0).astype(int)
    T = np.abs(labels[sorted_index])
    y_hat = preds[sorted_index]

    # segment index (a segment includes individuals with same observed time) 
    _t, seg_idx = np.unique(T, return_index=True)
    # seg_idx indicates the position of last one of each segment
    seg_idx = np.append(seg_idx[1:] - 1, n - 1)
    cnt_seg = len(seg_idx)

    # hazard ratio
    haz_ratio = np.exp(y_hat)
    cum_haz_ratio = np.cumsum(haz_ratio)
    sum_haz_ratio = cum_haz_ratio[-1]

    # SR(t)
    sr_t = sum_haz_ratio - np.append(0, cum_haz_ratio[seg_idx[:-1]])

    # segment event count
    cum_e = np.cumsum(E)
    cnt_e = cum_e[-1]
    seg_e = np.diff(np.append(0, cum_e[seg_idx]))

    # segment sum (hazard ratio of E=1)
    cum_dhr = np.cumsum(haz_ratio * E)
    seg_hdr = np.diff(np.append(0, cum_dhr[seg_idx]))

    # Compute Loss value (for each segment)
    loss = .0
    for i in np.arange(cnt_seg):
        if seg_e[i] > 0:
            w = np.arange(seg_e[i]).astype(np.float32) / seg_e[i]
            loss += np.sum(np.log(sr_t[i] - w * seg_hdr[i]))

    loss = loss - np.sum(E * y_hat)

    ### normalize by the number of events
    return "efn_loss", loss / cnt_e


def _efn_grads(preds, dtrain):
    """
    Gradient computation of custom objective function - Efron approximation.
    
    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by: 
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), 
        where N = #data.
    
    Returns
    -------
    tuple:
        The first- and second-order gradients of objective function w.r.t. `preds`.
    """
    # predictions
    n = preds.shape[0]
    
    # labels: get sorted E and T
    labels = dtrain.get_label()
    sorted_index = _label_abs_sort(labels)
    E = (labels[sorted_index] > 0).astype('int')
    T = np.abs(labels[sorted_index])
    y_hat = preds[sorted_index]

    # segment index (a segment includes individuals with same observed time) 
    _t, seg_idx = np.unique(T, return_index=True)
    # seg_idx indicates the position of last one of each segment
    seg_idx = np.append(seg_idx[1:] - 1, n - 1)
    cnt_seg = len(seg_idx)

    # hazard ratio
    haz_ratio = np.exp(y_hat)
    cum_haz_ratio = np.cumsum(haz_ratio)
    sum_haz_ratio = cum_haz_ratio[-1]

    # SR(t)
    sr_t = sum_haz_ratio - np.append(0, cum_haz_ratio[seg_idx[:-1]])

    # segment event count
    cum_e = np.cumsum(E)
    seg_e = np.diff(np.append(0, cum_e[seg_idx]))

    # segment sum (hazard ratio of E=1)
    cum_dhr = np.cumsum(haz_ratio * E)
    seg_hdr = np.diff(np.append(0, cum_dhr[seg_idx]))

    # Compute four pre-defined functions
    alpha = np.zeros_like(seg_idx, dtype=float)
    beta = np.zeros_like(seg_idx, dtype=float)
    phi = np.zeros_like(seg_idx, dtype=float)
    omega = np.zeros_like(seg_idx, dtype=float)
    for i in np.arange(cnt_seg):
        if seg_e[i] > 0:
            w = np.arange(seg_e[i]).astype(np.float32) / seg_e[i]
            contb = (sr_t[i] - w * seg_hdr[i])
            alpha[i] = np.sum(1.0 / contb)
            beta[i] = np.sum(w / contb)
            phi[i] = np.sum(1.0 / (contb ** 2))
            omega[i] = np.sum((1.0 - (1.0 - w) ** 2) / (contb ** 2))

    # Compute grads and hessian of each individual
    grad = np.zeros_like(y_hat)
    hess = np.zeros_like(y_hat)
    idx, sum_alpha, sum_phi = 0, .0, .0

    for seg in np.arange(cnt_seg):
        sum_alpha += alpha[seg]
        sum_phi += phi[seg]
        # for individuals in segment [idx, seg_idx[seg]]
        while idx <= seg_idx[seg]:
            g = haz_ratio[idx] * (sum_alpha - E[idx] * beta[seg]) - E[idx]
            h = g - (haz_ratio[idx] ** 2) * (sum_phi - E[idx] * omega[seg]) + E[idx]
            # filled in original order
            grad[sorted_index[idx]] = g
            hess[sorted_index[idx]] = h
            idx += 1

    return grad, hess
