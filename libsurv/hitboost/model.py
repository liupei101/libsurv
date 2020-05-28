"""HitBoost module
"""
import xgboost as xgb
import numpy as np

from ._hit_core import _global_init
from ._hit_core import _hit_grads
from ._hit_core import hit_tdci, hit_loss

from ._utils import _check_params, _check_data
from ._utils import _hit_eval
from ._utils import _print_eval

from ..vision import plot_train_curve, plot_surv_curve

class model(object):
    """HitBoost model"""
    def __init__(self, model_params=None, loss_alpha=1.0, loss_gamma=0.01, model_file=''):
        """
        HitBoost Model Constructor.

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

        loss_alpha: float
            The coefficient in objective function.
        loss_gamma: float
            The parameter in L2 term.
        model_file: str
            The model file path. This entry is mainly for loading the existing model.  

        Notes
        -----
        The type of objective must be `multi:softprob` and the 'num_class' is required.
        """
        super(model, self).__init__()
        # initialize global params
        _global_init(loss_alpha, loss_gamma)
        self.model_params = model_params
        self._model = None

        # loading the specified model
        if model_file != '':
            self._model = xgb.Booster(model_file=model_file)

    def train(self, dtrain, num_rounds=100, skip_rounds=10, evals=[], silent=False, plot=False):
        """
        HitBoost model training or watching learning curve on evaluation set.

        Parameters
        ----------
        dtrain: xgboost.DMatrix
            Training data for survival analysis. It's suggested that you utilize tools of 
            `datasets` module to convert pd.DataFrame to xgboost.DMatrix.
        num_rounds: int
            The number of iterations.
        skip_rounds: int
            The number of skipped rounds if you want to print infos.
        evals: list of pairs (xgb.DMatrix, string)
            Evaluation set to watch learning curve. If it is set as an empty list by default, 
            then the training data will became the evaluation set.
        silent: boolean
            Keep silence or print information.
        plot: boolean
            Plot learning curve.

        Returns
        -------
        dict:
            Evaluation result during training, which is formatted as `{'td-CI': [], 'Loss': []}`.
        """
        # First to check the args
        _check_params(self.model_params)
        # then check the train data
        _check_data(dtrain, self.model_params['num_class'])

        if len(evals) == 0:
            eval_labels = ['train']
            eval_datas = [dtrain]
        else:
            if not isinstance(evals[0], tuple):
                raise TypeError("The type of dtrain must be 'xgb.DMatrix'")
            eval_labels = [c[1] for c in evals]
            eval_datas = [c[0] for c in evals]
        
        # Logging for result
        eval_result = {'td-CI': [], 'Loss': []}
        self._model = xgb.Booster(self.model_params, [dtrain])
        for _ in range(num_rounds):
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
            # returns a list of values
            res_loss, res_ci = _hit_eval(self._model, eval_datas)
            eval_result['Loss'].append(res_loss)
            eval_result['td-CI'].append(res_ci)
            if not silent and (_ + 1) % skip_rounds == 0:
                _print_eval(_ + 1, res_loss, res_ci, eval_labels)

        # plot learning curve
        if plot:
            plot_train_curve(eval_result['Loss'], eval_labels, "Loss function")
            plot_train_curve(eval_result['td-CI'], eval_labels, "Time-Dependent C-index")

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
        numpy.ndarray
            prediction of P.D.F. of FHT with shape of `(N, k)`.
        """
        if not isinstance(ddata, xgb.DMatrix):
            raise TypeError("The type of data must be 'xgboost.DMatrix'")
        # Returns numpy.array
        preds = self._model.predict(ddata)
        return preds

    def predict_survival_function(self, ddata, plot=False):
        """
        Prediction method.

        Parameters
        ----------
        ddata: xgboost.DMatrix
            Test data for survival analysis. It's suggested that you utilize tools of 
            `datasets` module to convert pd.DataFrame to xgboost.DMatrix.
        plot: boolean
            Plot the predicted survival function.

        Returns
        -------
        numpy.ndarray
            prediction of survival function with shape of `(N, 1+k)`.
        """
        pred = self.predict(ddata)
        pred_surv = 1.0 - np.cumsum(pred, axis=1)
        pred_surv = np.insert(pred_surv, 0, values=np.ones(pred_surv.shape[0]), axis=1)

        # plot survival curve.
        if plot:
            plot_surv_curve(pred_surv)

        return pred_surv

    def evals(self, ddata):
        """
        evaluation performance of trained model on data.

        Parameters
        ----------
        ddata: xgboost.DMatrix
            Test data for survival analysis.

        Returns
        -------
        float
            Time-Dependent C-index.
        """
        preds = self.predict(ddata)
        return hit_tdci(preds, ddata)[1]

    def get_factor_score(self, importance_type='weight'):
        """
        Get the factor importance score evaluated by the model.
        It's suggested that you repeat obtaining the factor score 
        for multiply times, such as 20, by specifing a different random 
        seed in `model_params`. 

        Parameters
        ----------
        importance_type: str
            The metrics of importance evaluation. see more in [https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score].

        Returns
        -------
        dict
            Factor importance score.
        """
        return self._model.get_score(importance_type=importance_type)

    def save_model(self, file_path):
        """
        xgboost.Booster model saving.

        Parameters
        ----------
        file_path: str
            Path for local model saving.
        """
        # Model saving
        self._model.save_model(file_path)
