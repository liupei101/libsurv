import xgboost as xgb
import numpy as np

from ._hit_core import hit_tdci, hit_loss

EPS = 0.00001

def _check_params(model_params):
    """
    Check `model_params`.
    """
    if model_params is None:
        raise ValueError("If you want to train the model, you must \
            specify the parameter `model_params` when initializing the model.")

    if type(model_params) is not dict:
        raise TypeError("The type of `model_params` must be dict.")

    if "objective" in model_params:
        if model_params["objective"] != "multi:softprob":
            raise ValueError("The name of objective function must be 'multi:softprob'.")
    else:
        model_params["objective"] = "multi:softprob"

    if "num_class" not in model_params:
        raise ValueError("The parameter of 'num_class' must be included.")

def _check_data(data, params_num_class):
    """
    Check data type and validity.
    """
    if not isinstance(data, xgb.DMatrix):
        raise TypeError("The type of dtrain must be 'xgb.DMatrix'")

    
    y = data.get_label().astype('int')
    t = np.abs(y)

    # ensure that the time column of training data has been preprocessed to integer type. 
    if not (np.abs(y - data.get_label()) < EPS).all():
        raise ValueError("Float value found in the time column of training data.")

    # ensure that the time column of training data does not exist the value of zero.
    if (t == 0).any():
        raise ValueError("Zero found in the time column of training data.")

    # ensure that `params_num_class` included in `model_params` 
    # is equal to `K + 1`, where `K` denotes the maximum time in survival data.
    if params_num_class - 1 != np.max(t):
        raise ValueError("The value of 'num_class' in 'model_params' is \
            not equal to the maximum time plus one (i.e. K + 1) in train data")

def _hit_eval(model, eval_data):
    """
    Evaluate result on each iteration.

    Notes
    -----
    Only for `train` method of HitBoost. 
    """
    loss_list = []
    ci_list = []
    for d in eval_data:
        pred_d = model.predict(d)
        lossv = hit_loss(pred_d, d)[1]
        civ = hit_tdci(pred_d, d)[1]
        loss_list.append(lossv)
        ci_list.append(civ)
    return loss_list, ci_list

def _print_eval(iters_num, loss_list, ci_list, eval_labels):
    """
    Print evaluation result on each iteration.

    Notes
    -----
    Only for `train` method of HitBoost. 
    """
    print("# After %dth iteration:" % iters_num)
    for i in range(len(loss_list)):
        print("\tOn %s:" % eval_labels[i])
        print("\t\tLoss: %g" % loss_list[i])
        print("\t\ttd-CI: %g" % ci_list[i])