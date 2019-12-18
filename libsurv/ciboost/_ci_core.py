"""
L2 term of objective function in CEBoost.

CI approximated by convex function F and its gradients.

Convex function F = [-(y_hat[i] - y_hat[j] - _GAMMA)] ** 2
"""
import numpy as np

_GAMMA = 0.01

def ci_loss(preds, dtrain):
    """
    Computation of objective function.
    a.k.a. CI approximated by convex function.
    
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
    # predictions: np.array with shape of (N, )
    n = preds.shape[0]
    y_hat = preds

    # labels: np.array with shape of (N, )
    labels = dtrain.get_label().astype('int')
    E = (labels > 0).astype('int')
    T = np.abs(labels)

    # Compute the term of concordance index approximation
    loss_num = .0
    loss_den = .0
    for i in np.arange(n):
        if E[i] > 0:
            w = y_hat[i] - y_hat[T[i] < T]
            # For part of denominator and numerator
            loss_den += np.sum(-w)
            loss_num += np.sum((w < _GAMMA) * (-w) * (_GAMMA - w)**2)
    
    loss = 0 if loss_den == 0 else loss_num / loss_den

    return "ci_loss", loss

def _ci_grads(preds, dtrain):
    """
    Gradient computation of custom objective function.
    
    Parameters
    ----------
    preds: numpy.array
        An array with shape of (N, ), where N = #data.
    dtrain: xgboost.DMatrix
        Training data with type of `xgboost.DMatrix`. Labels can be obtained by: 
        `labels = dtrain.get_label()`, and it is `numpy.array` with shape of (N, ), 
        where N = #data. Negtive values are considered right censored (E = 0).

    Returns
    -------
    tuple:
        The first- and second-order gradients of objective function w.r.t. `preds`.
        
    Notes
    -----
    See notes of `hit_loss`.
    For efficiency, the implementation utilizes built-in function 
    in `numpy` as much as possible (maybe at the cost of readability).
    """
    # predictions: np.array with shape of (n, )
    n = preds.shape[0]
    y_hat = preds

    # labels: np.array with shape of (n, )
    labels = dtrain.get_label().astype('int')
    E = (labels > 0).astype('int')
    T = np.abs(labels)

    # L2 Gradient Computation (Concordance Index Approximation)
    # gradients computation of numerator and denominator in L2
    # initialization
    num, den = .0, .0
    grad_den = np.zeros_like(y_hat)
    hess_den = np.zeros_like(y_hat) # 0
    grad_num = np.zeros_like(y_hat)
    hess_num = np.zeros_like(y_hat)

    # firstly, compute gradients of numerator(\alpha) and denominator(\beta) in L2
    for k in np.arange(n):
        ## gradients of denominator (\beta)
        # For set s1 (i.e. \omega 1 in the paper)
        # s1 = (k, i): E_k = 1 and T_k < T_i
        s1 = E[k] * np.sum(T > T[k])
        # For set s2 (i.e. \omega 2 in the paper)
        # s2 = (i, k): E_i = 1 and T_i < T_k
        s2 = np.sum((E > 0) * (T < T[k]))
        # For grad_den (i.e. the first-order gradient of denominator)
        grad_den[k] = s2 - s1
        # hess_den[k] = 0

        ## gradients of numerator (\alpha)

        # set S1
        # i.e. the first-order and second-order gradients related to set s1
        # s1 = (k, i): E_k = 1 and T_k < T_i
        g_s1, h_s1 = .0, .0
        if E[k] == 1:
            w = y_hat[k] - y_hat[T[k] < T]
            # For den and num
            den += np.sum(-w)
            num += np.sum((w < _GAMMA) * (-w) * (_GAMMA - w)**2)

            g_s1 = np.sum((w < _GAMMA) * (_GAMMA - w) * (3*w - _GAMMA))

            h_s1 = np.sum((w < _GAMMA) * (4*_GAMMA - 6*w))
        
        # set S2
        # i.e. the first-order and second-order gradients related to set s2
        w = y_hat[(E > 0) * (T < T[k])] - y_hat[k]
        g_s2 = np.sum((w < _GAMMA) * (_GAMMA - w) * (_GAMMA - 3*w))
        h_s2 = np.sum((w < _GAMMA) * (4*_GAMMA - 6*w))
        
        grad_num[k] = g_s2 + g_s1
        hess_num[k] = h_s2 + h_s1

    if den == 0:
        grad_f = np.zeros_like(y_hat)
        hess_f = np.zeros_like(y_hat)
    else:
        grad_f = grad_num / den - num * grad_den / (den ** 2)
        hess_f = (den * hess_num - num * hess_den) / (den ** 2) - 2 * grad_den / den * grad_f
    
    return grad_f, hess_f
