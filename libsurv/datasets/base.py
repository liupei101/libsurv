"""Survival datasets preview or pre-processing module.
"""
from ..vision import plot_km_survf

def survival_stats(data, t_col="t", e_col="e", plot=False):
    """
    Print statistics of survival data to stdout.

    Parameters
    ----------
    data: pandas.DataFrame
        Survival data to watch.
    t_col: str
        Column name in data indicating time.
    e_col: str
        Column name in data indicating events or status.
    plot: boolean
        Is plot surival curve.
    """
    print("--------------- Survival Data Statistics ---------------")
    N = len(data)
    print("# Rows:", N)
    print("# Columns: %d + %s + %s" % (len(data.columns) - 2, e_col, t_col))
    print("# Events Ratio: %.2f%%" % (100.0 * data[e_col].sum() / N))
    print("# Min Time:", data[t_col].min())
    print("# Max Time:", data[t_col].max())
    print("")
    if plot:
        plot_km_survf(data, t_col="t", e_col="e")

def survival_df(data, t_col="t", e_col="e", label_col="Y", exclude_col=[], to_dmat=False):
    """
    Transform raw dataframe to survival dataframe that could be used in model 
    training or predicting (especially in XGBoost).

    Parameters
    ----------
    data: pandas.DataFrame
        Survival data to be transformed.
    t_col: str
        Column name in data indicating time.
    e_col: str
        Column name in data indicating events or status.
    label_col: str
        New label of transformed survival data.
    exclude_col: list
        Columns to be excluded.
    to_dmat: boolean
        Is transformed to xgboost.DMatrix.

    Returns
    -------
    DMatrix or DataFrame:
        Transformed survival data.
    """
    x_cols = [c for c in data.columns if c not in [t_col, e_col] + exclude_col]

    # Negtive values are considered right censored
    data.loc[:, label_col] = data.loc[:, t_col]
    data.loc[data[e_col] == 0, label_col] = - data.loc[data[e_col] == 0, label_col]

    # Returns X, Y
    if to_dmat:
        from xgboost import DMatrix
        mat_data = DMatrix(data[x_cols], label=data[label_col].values)
        return mat_data

    return data[x_cols + [label_col]]

def survival_dmat(data, **kwargs):
    """
    Survival data transformation for `DMatrix`. See more in `survival_df` function.
    """
    return survival_df(data, to_dmat=True, **kwargs)
