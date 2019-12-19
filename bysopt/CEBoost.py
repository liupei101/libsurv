"""
templates of tuning hyperparams for EfnBoost, read code and change anywhere if necessary.
The parts between two long strings "###..###" is more likely to be changed.
"""
# Set no warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
# load library
import sys
import time
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import hyperopt as hpt
from sklearn.model_selection import RepeatedKFold

from libsurv.datasets import survival_dmat
from libsurv import CEBoost

MSG = [
    "Error", # level 0
    "Warning", # level 1
    "Output", # level 2
    "Log", # level 3
    "Debug" # level 4
]

global collector, cur_round, skip_rounds, time_start
global train_data
global max_round, k_fold, n_repeats 
global col_t, col_e

def args_trans(args):
    """Parameters transform that must be specified."""
    params = {}
    #################################################################
    params["eta"] =  args["eta"] * 0.01 + 0.01
    params["nrounds"] =  args["nrounds"] * 10 + 80
    params['max_depth'] = 2 + args["max_depth"]
    params['min_child_weight'] = args['min_child_weight']
    params['subsample'] = args['subsample'] * 0.1 + 0.4
    params['colsample_bytree'] = args['colsample_bytree'] * 0.1 + 0.4
    params['reg_lambda'] = args['reg_lambda']
    params['reg_gamma'] = args['reg_gamma']
    params['loss_alpha'] = args['loss_alpha'] * 0.05 + 0.5
    #################################################################

    return params

def estimate_time():
    global time_start, cur_round
    
    time_now = time.clock()
    total = (time_now - time_start) / cur_round * (max_round - cur_round)
    th = int(total / 3600)
    tm = int((total - th * 3600) / 60)
    ts = int(total - th * 3600 - tm * 60)
    
    print("[%s] Estimate the remaining time: %dh %dm %ds" % (MSG[3], th, tm, ts))

def invoke_xgb(data_train, data_test, params):
    dtrain = survival_dmat(data_train, t_col=col_t, e_col=col_e, label_col="Y")
    dtest = survival_dmat(data_test, t_col=col_t, e_col=col_e, label_col="Y")
    
    # params
    params_model = {
        'eta': params['eta'],
        'max_depth': params['max_depth'], 
        'min_child_weight': params['min_child_weight'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'lambda': params['reg_lambda'],
        'gamma': params['reg_gamma'],
        'silent': 1
    }
    
    # Build and train model
    model = CEBoost(
        params_model,
        loss_alpha=params["loss_alpha"]
    )

    eval_result = model.train(
        dtrain,
        num_rounds=params['nrounds'],
        silent=True,
        plot=False
    )
    
    # Evaluation
    return model.evals(dtest)

def train_model(args):
    """"""
    global collector, cur_round, time_start
    global train_data

    params = args_trans(args)
    
    # Repeated KFold cross validation
    rskf = RepeatedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=64)
    metrics = []
    for train_index, test_index in rskf.split(train_data):
        data_train = train_data.iloc[train_index, :]
        data_validate = train_data.iloc[test_index,  :]
        metrics.append(invoke_xgb(data_train, data_validate, params))
    metrics_mean = np.array(metrics).mean()
    
    # Write log
    collector.append({'params': params, 'ci': metrics_mean})
    cur_round += 1
    
    # Estimate time left
    if cur_round % skip_rounds == 0:
        print("[%s] At %d-th round:" % (MSG[3], cur_round), end=" ") 
        print(params, metrics_mean)
        estimate_time()
    
    return 1.0 - metrics_mean

def search_params(max_evals=100):
    ###################################################################
    # Parameters' space
    space = {
        "eta": hpt.hp.randint("eta", 10),  # [0.01, 0.10] = 0.01 * [0, 9] + 0.01
        "nrounds": hpt.hp.randint("nrounds", 8),  # [80, 150] = 10 * [0, 7] + 80
        "max_depth": hpt.hp.randint("max_depth", 5), # [2, 6] = [0, 4] + 2
        "min_child_weight": hpt.hp.uniform("min_child_weight", 0.0, 1.0), # [0.0, 1.0]
        "subsample": hpt.hp.randint("subsample", 7), # [0.4, 1.0] = 0.1 * [0, 6] + 0.5
        "colsample_bytree": hpt.hp.randint("colsample_bytree", 7),# [0.4, 1.0] = 0.1 * [0, 6] + 0.5
        "reg_lambda" : hpt.hp.uniform("reg_lambda", 0.0, 1.0), # [0.0, 1.0]
        "reg_gamma":  hpt.hp.uniform("reg_gamma", 0.0, 1.0), # [0.0, 1.0]
        "loss_alpha": hpt.hp.randint("loss_alpha", 11),  # [0.5, 1.0] = 0.05 * [0, 10] + 0.5
    }

    ####################################################################
    # Minimize
    best = hpt.fmin(train_model, space, algo=hpt.tpe.suggest, max_evals=max_evals)
    print("[%s] Hyper-parameters searching finished!" % MSG[3])
    print("[%s] Best params :" % MSG[2], args_trans(best))
    print("[%s] Best metrics:" % MSG[2], 1.0 - train_model(best))

def write_file(message, filepath):
    """Write message into the specified file formatted as JSON"""
    with open(filepath, 'w') as f:
        json.dump(message, f)


##############################################################
# Usage: python3 filepath_input filepath_output
# Note: file path of train dataset: "filepath_input", 
#       such as "whas_train.csv"
# Note: file path of parameters tuning log: "filepath_output", 
#       such as "whas.json"
if __name__ == "__main__":
    # File name of input and output
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    train_data = pd.read_csv(input_file)
    print("[%s] No. Rows:" % MSG[2], len(train_data))
    print("[%s] ID. Cols:" % MSG[2], train_data.columns)

    ##########################################
    # Initialize global variables
    collector = []
    cur_round = 0
    skip_rounds = 1
    max_round = 200
    k_fold = 10
    n_repeats = 3
    col_t = 't'
    col_e = 'e'
    time_start = time.clock()
    ##########################################
    
    search_params(max_evals=max_round)
    write_file(collector, output_file)