import numpy as np
import pandas as pd
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

def plot_surv_curve(df_survf, title="Survival Curve"):
    """
    Plot survival curve.

    Parameters
    ----------
    df_survf: DataFrame
        Survival function of samples, shape of which is (n, #Time_Points).
        `Time_Points` indicates the time point presented in columns of DataFrame.
    """
    plt.plot(df_survf.columns.values, np.transpose(df_survf.values))
    plt.title(title)
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
    e = (y_true > 0).astype(np.int32)
    ci_value = ci(t, -y_pred, e)
    return ci_value

def baseline_hazard_(label_e, label_t, pred_hr):
    ind_df = pd.DataFrame({"E": label_e, "T": label_t, "P": pred_hr})
    summed_over_durations = ind_df.groupby("T")[["P", "E"]].sum()
    summed_over_durations["P"] = summed_over_durations["P"].loc[::-1].cumsum()
    # where the index of base_haz is sorted time from small to large
    # and the column `base_haz` is baseline hazard rate
    base_haz = pd.DataFrame(
        summed_over_durations["E"] / summed_over_durations["P"], columns=["base_haz"]
    )
    return base_haz

def baseline_cumulative_hazard_(label_e, label_t, pred_hr):
    return baseline_hazard_(label_e, label_t, pred_hr).cumsum()

def baseline_survival_function_(label_e, label_t, pred_hr):
    base_cum_haz = baseline_cumulative_hazard_(label_e, label_t, pred_hr)
    survival_df = np.exp(-base_cum_haz)
    return survival_df

def baseline_survival_function(y, pred_hr):
    """
    Estimate baseline survival function by Breslow Estimation.

    Parameters
    ----------
    y : np.array
        Observed time. Negtive values are considered right censored.
    pred_hr : np.array
        Predicted value, i.e. hazard ratio.

    Returns
    -------
    DataFrame
        Estimated baseline survival function. Index of it is time point. 
        The only one column of it is corresponding survival probability.
    """
    y = np.squeeze(y)
    pred_hr = np.squeeze(pred_hr)
    # unpack label
    t = np.abs(y)
    e = (y > 0).astype(np.int32)
    return baseline_survival_function_(e, t, pred_hr)