import numpy as np
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index as ci

def plot_train_curve(L, title='Training Curve'):
    if type(L) == list:
        x = range(1, len(L) + 1)
        plt.plot(x, L, label="evaluation set")
    elif type(L) == dict:
        for k, v in L.items():
            x = range(1, len(v) + 1)
            plt.plot(x, v, label=k)
    # no ticks
    plt.xticks([])
    plt.legend(loc="best")
    plt.title(title)
    plt.show()

def plot_surv_func(T, surRates):
    plt.plot(T, np.transpose(surRates))
    plt.show()

def concordance_index(self, y_true, y_pred):
    """
    Compute the concordance-index value.

    Parameters
    ----------
    y_true : np.array
        Observed time. Negtive values are considered right censored.
    y_pred : np.array
        Predicted value.

    Returns
    -------
    float
        Concordance index.
    """
    t = np.abs(y_true)
    e = (y_true > 0).astype(np.int)
    ci_value = ci(t, -y_pred, e)
    return ci_value