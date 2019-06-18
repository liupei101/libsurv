"""HitBoost module
"""
import xgboost as xgb
from ._hit_core import *

def _hit_eval(model, eval_data=[]):
    """
    Evaluate result on each iteration.

    Notes
    -----
    Only for `learning_curve` method of HitBoost. 
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

def _print_eval(iters_num, loss_list, ci_list):
    """
    Print evaluation result on each iteration.

    Notes
    -----
    Only for `learning_curve` method of HitBoost. 
    """
    print("# After %dth iteration:" % iters_num)
    for i in range(len(loss_list)):
        print("\tOn %d-th data:" % (i + 1))
        print("\t\tLoss: %g" % loss_list[i])
        print("\t\ttd-CI: %g" % ci_list[i])

class model(object):
    """HitBoost model"""
    def __init__(self, model_params, num_rounds=100, 
             loss_alpha=1.0, loss_gamma=0.01):
        """
        Class initialization.

        Parameters
        ----------
        model_params: dict
            Parameters of xgboost multi-classification model.
            See more in `Reference <https://xgboost.readthedocs.io/en/latest/parameter.html>`.

            For example:
                params = 
                {
                    'eta': 0.1,
                    'max_depth':3, 
                    'min_child_weight': 8, 
                    'subsample': 0.9,
                    'colsample_bytree': 0.5,
                    'gamma': 0,
                    'lambda': 0,
                    'silent': 1,
                    'objective': 'multi:softprob',
                    'num_class': K+1,
                    'seed': 42
                }

        num_rounds: int
            The number of iterations.

        loss_alpha: float
            The coefficient in objective function.

        loss_gamma: float
            The parameter in L2 term.

        Notes
        -----
        The type of objective must be `multi:softprob` and 'num_class' is necessary.
        """
        super(model, self).__init__()
        # initialize global params
        _global_init(loss_alpha, loss_gamma)
        self.model_params = model_params
        self.num_rounds = num_rounds
        self._model = None

    def _check_params(self):
        """
        Check `model_params` and raise errors.
        """
        if "objective" in self.model_params:
            if self.model_params["objective"] != "multi:softprob":
                raise ValueError("The name of objective function must be 'multi:softprob'.")
        else:
            self.model_params["objective"] = "multi:softprob"

        if "num_class" not in self.model_params:
            raise ValueError("The parameter of 'num_class' must be included.")

    def train(self, dtrain):
        """
        HitBoost model training.

        Parameters
        ----------
        dtrain: xgboost.DMatrix
            Training data for survival analysis. It's suggested that you utilize tools of 
            `datasets` module to convert pd.DataFrame to xgboost.DMatrix.
        """
        # Firstly check the args
        _check_params()
        # Is DMatrix
        if not isinstance(dtrain, xgb.DMatrix):
            raise TypeError("The type of dtrain must be 'xgb.DMatrix'")

        self._model = xgb.Booster(self.model_params, [dtrain])
        for _ in range(self.num_rounds):
            # Note: Since default setting of `output_margin` is `False`,
            # so the prediction is outputted after softmax transformation.
            pred = self._model.predict(dtrain)
            # Note: The gradient you provide for `model.boost()` must be 
            # gradients of objective function with respect to the direct 
            # output of boosting tree (even if you set `output_margin` as 
            # `True`).
            g, h = _hit_grads(pred, dtrain)
            self._model.boost(dtrain, g, h)

    def learning_curve(self, dtrain, eval_data, silent=True):
        """
        HitBoost model training and watching learning curve on evaluation set.

        Parameters
        ----------
        dtrain: xgboost.DMatrix
            Training data for survival analysis. It's suggested that you utilize tools of 
            `datasets` module to convert pd.DataFrame to xgboost.DMatrix.

        eval_data: list
            Evaluation set to watch learning curve.

        silent: boolean
            Keep silence or print information.

        Returns
        -------
        dict:
            Evaluation result during training, which is formatted as `{'td-CI': [], 'Loss': []}`.
        """
        # Firstly check the args
        _check_params()
        # Is DMatrix
        if not isinstance(dtrain, xgb.DMatrix):
            raise TypeError("The type of dtrain must be 'xgb.DMatrix'")
        # Logging for result
        eval_result = {'td-CI': [], 'Loss': []}
        self._model = xgb.Booster(self.model_params, [dtrain])
        for _ in range(self.num_rounds):
            # Note: Since default setting of `output_margin` is `False`,
            # so the prediction is outputted after softmax transformation.
            pred = self._model.predict(dtrain)
            # Note: The gradient you provide for `model.boost()` must be 
            # gradients of objective function with respect to the direct 
            # output of boosting tree (even if you set `output_margin` as 
            # `True`).
            g, h = _hit_grads(pred, dtrain)
            self._model.boost(dtrain, g, h)
            # Append to eval_result
            if len(eval_data) > 0:
                res_loss, res_ci = _hit_eval(model, eval_data)
                eval_result['Loss'].append(res_loss)
                eval_result['td-CI'].append(res_ci)
                if not silent:
                    _print_eval(_ + 1, res_loss, res_ci)

        return eval_result

    def predict(self, ddata):
        """
        Prediction method.

        Parameters
        ----------
        ddata: xgboost.DMatrix
            Test data for survival analysis. It's suggested that you utilize tools of 
            `datasets` module to convert pd.DataFrame to xgboost.DMatrix.

        Returns
        -------
        numpy.array
            prediction with shape of `(N, k)`.
        """
        if not isinstance(ddata, xgb.DMatrix):
            raise TypeError("The type of dtrain must be 'xgb.DMatrix'")
        # Returns numpy.array
        preds = self._model.predict(ddata)
        return preds

    def save_model(self, file_path):
        """
        Model saving.

        Parameters
        ----------
        file_path: str
            Path for local model saving.
        """
        # Model saving
        self._model.save_model(file_path)
