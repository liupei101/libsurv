# coding=utf-8
# Set no warning
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
# load library
import time
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import hyperopt as hpt
from sklearn.model_selection import RepeatedKFold

from libsurv.datasets import survival_dmat
from libsurv import EfnBoost

global Logval, eval_cnt, time_start
global train_data
global max_iters, k_fold, n_repeats 
global T_col, E_col

def args_trans(args):
    params = {}
    params["eta"] =  args["eta"] * 0.01 + 0.01
    params["nrounds"] =  args["nrounds"] * 10 + 80
    params['max_depth'] = 2 + args["max_depth"]
    params['min_child_weight'] = args['min_child_weight']
    params['subsample'] = args['subsample'] * 0.1 + 0.4
    params['colsample_bytree'] = args['colsample_bytree'] * 0.1 + 0.4
    params['reg_lambda'] = args['reg_lambda']
    params['reg_gamma'] = args['reg_gamma']
    return params

def estimate_time():
    global time_start, eval_cnt
    time_now = time.clock()
    total = (time_now - time_start) / eval_cnt * (max_iters - eval_cnt)
    th = int(total / 3600)
    tm = int((total - th * 3600) / 60)
    ts = int(total - th * 3600 - tm * 60)
    print 'Estimate the remaining time: %dh %dm %ds' % (th, tm, ts)

def invoke_xgb(data_train, data_test, params):
    dtrain = survival_dmat(data_train, t_col=T_col, e_col=E_col, label_col="Y")
    dtest = survival_dmat(data_test, t_col=T_col, e_col=E_col, label_col="Y")
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
    model = EfnBoost(params_model)
    eval_result = model.train(
        dtrain,
        num_rounds=params['nrounds'],
        silent=True,
        plot=False
    )
    # Evaluation
    return model.evals(dtest)

def train_model(args):
    global Logval, eval_cnt, time_start
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
    Logval.append({'params': params, 'ci': 1-metrics_mean})
    eval_cnt += 1
    # Estimate time left
    if eval_cnt % 1 == 0:
        print params, 1-metrics_mean
        estimate_time()
    
    return metrics_mean

def search_params(max_evals=100):
    global Logval
    ###################################################################
    # Parameters' space
    space = {
        "eta": hpt.hp.randint("eta", 10),  # [0.01, 0.10] = 0.01 * [0, 9] + 0.01
        "nrounds": hpt.hp.randint("nrounds", 8),  # [80, 150] = 10 * [0, 7] + 80
        "max_depth": hpt.hp.randint("max_depth", 5), # [2, 6] = [0, 4] + 2
        "reg_gamma":  hpt.hp.uniform("reg_gamma", 0.0, 1.0), # [0.0, 1.0]
        "min_child_weight": hpt.hp.uniform("min_child_weight", 0.0, 1.0), # [0.0, 1.0]
        "subsample": hpt.hp.randint("subsample", 7), # [0.4, 1.0] = 0.1 * [0, 6] + 0.5
        "colsample_bytree": hpt.hp.randint("colsample_bytree", 7),# [0.4, 1.0] = 0.1 * [0, 6] + 0.5
        "reg_lambda" : hpt.hp.uniform("reg_lambda", 0.0, 1.0) # [0.0, 1.0]
    }
    ####################################################################
    # Minimize
    best = hpt.fmin(train_model, space, algo=hpt.tpe.suggest, max_evals=max_evals)
    print "Hyperopt Running Finished !"
    print "\tBest params :", args_trans(best)
    print "\tBest metrics:", train_model(best)

def write_file(filename, var):
    with open(filename, 'w') as f:
        json.dump(var, f)

# Usage:
# - python3 input_file_path output_file_path
# - input_file_path: whas_train.csv
# - output_file_path: whas.json
if __name__ == "__main__":
    #### Set file name of input and output ###
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    ##########################################
    train_data = pd.read_csv(input_file)
    print "No. Rows:", len(train_data)
    print "ID. Cols:", train_data.columns
    ##########################################
    ###    Initialize global variables     ###
    Logval = []
    eval_cnt = 0
    max_iters = 200
    k_fold = 10
    n_repeats = 3
    T_col = 'dp_month'
    E_col = 'dp_bin'
    time_start = time.clock()
    ##########################################
    search_params(max_evals=max_iters)
    write_file(output_file, Logval)