"""BecCox module
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from ..vision import plot_train_curve, plot_surv_curve
from ..utils import concordance_index, baseline_survival_function, _check_params

from ._core import _params_init
from ._core import _ce_grads, ce_loss, ce_evals


class model(object):
    """BecCox model class"""
    def __init__(self, model_params=None, loss_alpha=.0, model_file=None):
        """
        BecCox Class Constructor.

        Parameters
        ----------
        model_params: dict
            Parameters of `xgboost.train` method.
            See more in `Reference <https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training>`.
        
        model_file: str
            The model file path. This entry is mainly for loading the existing model.  
        """
        super(model, self).__init__()

        _params_init(loss_alpha)
        self.model_params = model_params
        self._model = None

        # loading the specified model
        self.model_file = model_file
        if model_file is not None:
            self._model = xgb.Booster(model_file=model_file)

    def train(self, dtrain, num_rounds=100, skip_rounds=10, evals=[], name_evals="ce_evals", silent=False, plot=False):
        """
        BecCox model training and watching learning curve on evaluation set.

        Parameters
        ----------
        dtrain: xgboost.DMatrix
            Training data for survival analysis. It's suggested that you utilize tools of 
            `datasets` module to convert pd.DataFrame to xgboost.DMatrix.
        num_rounds: int
            The number of iterations.
        skip_rounds: int or boolean
            The number of skipped rounds if you want to print infos.
            Requires at least one item in evals. If verbose_eval is True then the evaluation 
            metric on the validation set is printed at each boosting stage. If verbose_eval 
            is an integer then the evaluation metric on the validation set is printed at every 
            given verbose_eval boosting stage. The last boosting stage / the boosting stage found 
            by using early_stopping_rounds is also printed. Example: with verbose_eval=4 and at 
            least one item in evals, an evaluation metric is printed every 4 boosting stages, 
            instead of every boosting stage.
        evals: list of pairs (xgb.DMatrix, string)
            Evaluation set to watch learning curve. If it is set as an empty list by default, 
            then the training data will became the evaluation set.
        name_evals: string
            The name of evaluation method. Only 'ce_loss' and 'ce_evals' are available.
        silent: boolean
            Print infos to screen.
        plot: boolean
            Plot the learning curve.

        Returns
        -------
        dict
            This dictionary stores the evaluation results of all the items in watchlist.
            Example: with a watchlist containing [(dtest,'eval'), (dtrain,'train')] and 
            a parameter containing ('eval_metric': 'logloss'), the evals_result returns
            {'train': {'efn_loss': ['0.48253', '0.35953']},
             'eval': {'efn_loss': ['0.480385', '0.357756']}}
        """
        # First to check the args
        _check_params(self.model_params)
        # Check arguements of function
        if not isinstance(dtrain, xgb.DMatrix):
            raise TypeError("The type of dtrain must be 'xgb.DMatrix'")

        if len(evals) == 0:
            evals = [(dtrain, 'train')]
        
        if silent:
            skip_rounds = False

        if name_evals == "ce_loss":
            func_evals = ce_loss
        elif name_evals == "ce_evals":
            func_evals = ce_evals
        else:
            raise NotImplementedError("Can not recognize evaluation method.")

        # Train model
        evals_result = {}
        self._model = xgb.train(
            self.model_params, 
            dtrain, 
            num_boost_round=num_rounds,
            evals=evals,
            evals_result=evals_result,
            obj=_ce_grads,
            feval=func_evals,
            verbose_eval=skip_rounds,
            xgb_model=self.model_file
        )

        if plot:
            # data format transform
            data_labels = [c[1] for c in evals]
            _evals_list = []
            for c in data_labels:
                _evals_list.append(evals_result[c][name_evals])
            evals_zipped = list(zip(*_evals_list))
            # plot
            plot_train_curve(evals_zipped, data_labels, "Performance")

        # update the baseline survival function after training
        self.HR = self.predict(dtrain, output_margin=False)
        # we estimate the baseline survival function S0(t) using training data
        # which returns a DataFrame
        self.BSF = baseline_survival_function(dtrain.get_label(), self.HR)

        return evals_result

    def predict(self, ddata, output_margin=True):
        """
        Prediction method.
        
        Parameters
        ----------
        ddata: xgboost.DMatrix
            Test data for survival analysis. It's suggested that you utilize tools of 
            `datasets` module to convert pd.DataFrame to xgboost.DMatrix.
        output_margin: boolean
            If output_margin is set to True, then output of model is log hazard ratio.
            Otherwise the output is hazard ratio, i.e. exp(beta*x).
        
        Returns
        -------
        numpy.array
            prediction with shape of `(N, )` indicting predicted hazard ratio.
        """
        if not isinstance(ddata, xgb.DMatrix):
            raise TypeError("The type of data must be 'xgb.DMatrix'")
        # Make prediction
        preds = self._model.predict(ddata)
        if output_margin:
            return preds
        return np.exp(preds)

    def predict_survival_function(self, X, plot=False):
        """
        Predict survival function of samples.
        
        Parameters
        ----------
        X: xgboost.DMatrix
            Input data.
        plot: boolean
            Plot the estimated survival curve of samples.
        
        Returns
        -------
        pandas.DataFrame
            Predicted survival function of samples, shape of which is (n, #Time_Points).
            `Time_Points` indicates the time point that exists in the training data.
        """
        pred_hr = self.predict(X, output_margin=False)
        pred_hr = pred_hr.reshape((pred_hr.shape[0], 1))
        survf = pd.DataFrame(self.BSF.iloc[:, 0].values ** pred_hr, columns=self.BSF.index.values)
        
        # plot survival curve
        if plot:
            plot_surv_curve(survf)

        return survf

    def evals(self, dtest):
        """
        Evaluate labeled dataset using the CI metrics under current trained model.
        
        Parameters
        ----------
        dtest: xgboost.DMatrix
            Test data for survival analysis.
        
        Returns
        -------
        float
            CI metrics on your dataset.
        
        Notes
        -----
        We use negtive hazard ratio as the score. See https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
        """
        score = -self.predict(dtest)
        return concordance_index(dtest.get_label(), score)

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