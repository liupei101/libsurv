"""EfnBoost model
"""
import xgboost as xgb
from ._efn_core import *

class model(object):
    """EfnBoost model class"""
    def __init__(self, model_params, num_rounds):
        """
        Class initialization.

        Parameters
        ----------
        model_params: dict
            Parameters of `xgboost.train` method.
            See more in `Reference <https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training>`.

        num_rounds: int
            The number of iterations.
        """
        super(model, self).__init__()
        self.model_params = model_params
        self.num_rounds = num_rounds
        self._model = None

    def train(self, dtrain, evals=[], evals_result=None):
        """
        EfnBoost model training and watching learning curve on evaluation set.

        Parameters
        ----------
        dtrain: xgboost.DMatrix
            Training data for survival analysis. It's suggested that you utilize tools of 
            `datasets` module to convert pd.DataFrame to xgboost.DMatrix.

        evals: list
            Evaluation set to watch learning curve.

        evals_result: dict
            Store the result of evaluation.
        """
        if not isinstance(dtrain, xgb.DMatrix):
            raise TypeError("The type of dtrain must be 'xgb.DMatrix'")
        # Train model
        if evals_result is None:
            # No evaluation set
            self._model = xgb.train(
                self.model_params, 
                dtrain, 
                num_boost_round=self.num_rounds,
                obj=_efn_grads,
                feval=efn_loss
            )
        else:
            # If given the evaluation set
            self._model = xgb.train(
                self.model_params, 
                dtrain, 
                num_boost_round=self.num_rounds,
                evals=evals,
                evals_result=evals_result,
                obj=_efn_grads,
                feval=efn_loss
            )

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
            prediction with shape of `(N, )` indicting predicted hazard ratio.
        """
        if not isinstance(ddata, xgb.DMatrix):
            raise TypeError("The type of data must be 'xgb.DMatrix'")
        # Make prediction
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
