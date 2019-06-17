"""Custom loss function and corresponding gradients in HitBoost model.

----------
Parameters
----------

preds
-----
numpy.array with shape of (N, K), where N = #data, K = #classes.
K denotes the number of class, which is declared in the parameter "num_class" 
of XGBoost model.
In this application, K denotes the number of different time point, and `preds` 
denotes the probability distribution of time when individual i occurs event.

dtrain
------
DMatrix object with training data. labels can be obtained by: 
labels = dtrain.get_label(), and labels is np.array with shape of (N, ), where
N = #data.

In this way, absolute value of label represents T in survival data, 
Negtive values are considered right censored, i.e. E = 0; Positive values are 
considered event occurrence, i.e. E = 1.
"""
import numpy as np

# Coefficient of CI term in objective function.
global _theta
# Hyper-parameter in CI term
global _gamma

def _global_init(tval, gval):
    """
    initialization of global variables.
    """
    global _gamma, _theta
    _theta = tval
    _gamma = gval

def hit_loss(preds, dtrain):
    """
    Computation of Objective Function
    """
    # np.array with shape of (N, k)
    yhat = preds
    N, K = yhat.shape
    # np.array with shape of (N, )
    y = dtrain.get_label().astype('int')
    t = np.abs(y)
    # L1 Computation (term of likelihood function)
    L1 = .0
    for i in range(N):
        if y[i] > 0:
            L1 = L1 - np.log(yhat[i, t[i]-1])
        else:
            L1 = L1 - np.log(1.0 - np.sum(yhat[i, :t[i]]))
    L1 /= N
    # L2 Computation (term of concordance index approximation)
    L2_num = .0
    L2_den = .0
    for i in range(N):
        if y[i] > 0:
            p = np.sum(yhat[i, :t[i]]) - np.sum(yhat[t[i] < t, :t[i]], axis=1, keepdims=True)
            # For part of denominator and numerator
            L2_den += -np.sum(p)
            L2_num += -np.sum((p < _gamma) * p * (_gamma - p)**2)
    L2 = 0 if L2_den == 0 else L2_num / L2_den
    # L Computation (L = _theta * L1 + (1 - _theta) * L2)
    return "Loss", _theta * L1 + (1 - _theta) * L2

def hit_tdci(preds, dtrain):
    """
    Computation of Time-Dependent Concordance Index
    """
    # np.array with shape of (N, k)
    yhat = preds
    N, K = yhat.shape
    # np.array with shape of (N, )
    y = dtrain.get_label().astype('int')
    t = np.abs(y)
    # state the count variables
    Nsum = 0
    Ny = 0
    for i in range(N):
        if y[i] > 0:
            p = np.sum(yhat[i, :t[i]]) - np.sum(yhat[t[i] < t, :t[i]], axis=1, keepdims=True)
            Nsum += p.shape[0]
            Ny += np.sum(p > 0)
    return "td-CI", 1.0 * Ny / Nsum

def _hit_grads(preds, dtrain):
    """
    Gradient Computation of custom objective function.
    For high speed running, the implementation utilizes built-in function 
    in `numpy` as much as possible (at the cost of readability).
    """
    # np.array with shape of (N, k)
    yhat = preds
    N, K = yhat.shape
    # np.array with shape of (N, )
    y = dtrain.get_label().astype('int')
    t = np.abs(y)

    # individuals of right-censoring and non-right-censoring
    evto = y > 0
    evto_t = t[evto] - 1
    evto_no = len(evto_t)
    cens = y < 0
    cens_t = t[cens] - 1
    cens_no = N - evto_no

    # L1 Gradient Computation (likelihood function)
    # initialization
    L1_grad = np.zeros_like(yhat)
    L1_hess = np.zeros_like(yhat)
    ## For events individuals
    L1_grad[evto, evto_t] = -1.0 / yhat[evto, evto_t]
    L1_hess[evto, evto_t] = 1.0 / (yhat[evto, evto_t]**2)
    ## For censoring individuals
    cif = np.cumsum(yhat[cens], axis=1)[np.arange(cens_no), cens_t].reshape(-1, 1)
    cens_mask = cens_t[:, None] >= np.arange(K)
    L1_grad[cens, :] = 1.0 / (1.0 - cif)
    L1_grad[cens, :] = L1_grad[cens, :] * cens_mask
    L1_hess[cens, :] = 1.0 / ((1.0 - cif)**2)
    L1_hess[cens, :] = L1_hess[cens, :] * cens_mask

    # L2 Gradient Computation (Concordance Index Approximation)
    # gradients computation of numerator and denominator in L2
    # initialization
    num, den = .0, .0
    grad_num = np.zeros_like(yhat)
    hess_num = np.zeros_like(yhat)
    grad_den = np.zeros_like(yhat)
    hess_den = np.zeros_like(yhat) # 0
    # firstly, compute gradients of numerator(\alpha) and denominator(\beta) in L2
    for k in range(N):
        s1, s2 = np.zeros_like(yhat[0]), np.zeros_like(yhat[0])
        g_s1, g_s2 = np.zeros_like(yhat[0]), np.zeros_like(yhat[0])
        h_s1, h_s2 = np.zeros_like(yhat[0]), np.zeros_like(yhat[0])
        # For set s1 (i.e. \omega 1 in the paper)
        s1[:t[k]] = np.sum(t > t[k]) * (y[k] > 0)
        # For set s2 (i.e. \omega 2 in the paper)
        hev_t = t[(t < t[k]) * (y > 0)]
        mask_s2 = hev_t[:, None] > np.arange(K)
        s2 = np.sum(mask_s2, axis=0)
        # For grad_den (i.e. the first-order gradient of denominator)
        grad_den[k] = s2 - s1
        if y[k] > 0:
            p = np.sum(yhat[k, :t[k]]) - np.sum(yhat[t[k] < t, :t[k]], axis=1, keepdims=True)
            # For den and num
            den += -np.sum(p)
            num += -np.sum((p < _gamma) * p * (_gamma - p)**2)
            # For g_s1 and h_s1 
            # i.e. the first-order and second-order gradients related to set s1
            g_s1[:t[k]] = np.sum((p < _gamma) * (_gamma - p) * (3*p - _gamma))
            h_s1[:t[k]] = np.sum((p < _gamma) * (4*_gamma - 6*p))
        # For g_s2 and h_s2
        # i.e. the first-order and second-order gradients related to set s2
        p = np.sum(yhat[(y > 0) * (t < t[k])] * mask_s2, axis=1, keepdims=True) - \
                np.sum(yhat[k] * mask_s2, axis=1, keepdims=True)
        g_s2 = np.sum((p < _gamma) * (_gamma - p) * (_gamma - 3*p) * mask_s2, axis=0)
        h_s2 = np.sum((p < _gamma) * (4*_gamma - 6*p) * mask_s2, axis=0)
        grad_num[k] = g_s2 + g_s1
        hess_num[k] = h_s2 + h_s1
    # L2 Gradient Computation
    if den == 0:
        L2_grad = np.zeros_like(yhat)
        L2_hess = np.zeros_like(yhat)
    else:
        L2_grad = (den * grad_num - num * grad_den) / (den**2)
        L2_hess = (den * hess_num - num * hess_den) / (den**2) - 2 * grad_den / den * L2_grad
    # L Gradient Computation
    # L = _theta * L1 + (1 - _theta) * L2
    grad = _theta * L1_grad + (1 - _theta) * L2_grad
    hess = _theta * L1_hess + (1 - _theta) * L2_hess
    # NOTE: the output of XGBoost has been tranformed by softmax.
    #       But the gradient we compute must be with respect to the 
    #       predicted values before softmax transformation.
    # Gradient before softmax layer
    grad_f = yhat * (1.0 - yhat) * grad
    hess_f = yhat * (1.0 - yhat) * (grad * (1.0 - 2 * yhat) + hess * yhat * (1.0 - yhat))
    return grad_f.flatten(), hess_f.flatten()