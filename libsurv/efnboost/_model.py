import xgboost as xgb
from ._efn_core import *

class model(object):
    """docstring for model"""
    def __init__(self, model_params, num_rounds):
        super(model, self).__init__()
        self.model_params = model_params
        self.num_rounds = num_rounds
        self._model = None

    def train(self, dtrain, evals, evals_result=None):
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
        if not isinstance(ddata, xgb.DMatrix):
            raise TypeError("The type of data must be 'xgb.DMatrix'")
        # Make prediction
        preds = self._model.predict(ddata)
        return preds

    def save_model(self, file_path):
        # Model saving
        self._model.save_model(file_path)
