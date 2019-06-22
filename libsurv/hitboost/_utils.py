from ._hit_core import hit_tdci, hit_loss

def _check_params(model_params):
    """
    Check `model_params` and raise errors.
    """
    if "objective" in model_params:
        if model_params["objective"] != "multi:softprob":
            raise ValueError("The name of objective function must be 'multi:softprob'.")
    else:
        model_params["objective"] = "multi:softprob"

    if "num_class" not in model_params:
        raise ValueError("The parameter of 'num_class' must be included.")

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