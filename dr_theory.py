# dr_theory.py
import os
import numpy as np
import theano
import theano.tensor as T

from lib.utils.theano_utils import *
from lib.utils.lasagne_utils import *
from lib.utils.data_utils import *
from lib.utils.dr_utils import *
from lib.utils.attack_utils import *
from lib.utils.plot_utils import *
from lib.utils.model_utils import *


def gradient_calc(rd, model_dict, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """
    Robust gradient calculation. Returns list of (feature_std, avg_gradient) pairs.
    Handles optional validation set and varying shapes from model_setup.
    """

    rev_flag = None
    dim_red = model_dict.get('dim_red', None)

    # model_setup may modify / return a different X_test; prefer that returned one
    data = model_setup(model_dict, X_train, y_train, X_test, y_test, X_val, y_val, rd, rev=rev_flag)
    # Unpack defensively
    if len(data) >= 6:
        data_dict, test_prediction, dr_alg, X_test_model, input_var, target_var = data[:6]
    else:
        raise ValueError("model_setup returned unexpected number of values: {}".format(len(data)))

    # Prefer returned X_test if provided
    X_test_used = X_test_model if X_test_model is not None else X_test
    X_test_used = np.asarray(X_test_used)

    # Ensure 2D (n_samples, n_features)
    if X_test_used.ndim > 2:
        X_test_used = X_test_used.reshape((X_test_used.shape[0], -1))

    # Get test_len and no_of_features from data_dict when possible
    test_len = data_dict.get('test_len', X_test_used.shape[0])
    no_of_features = data_dict.get('no_of_features', X_test_used.shape[1])

    # Try to reshape safely
    try:
        X_test_dr = X_test_used.reshape((test_len, no_of_features))
    except Exception:
        # fallback: use whatever shape we have
        X_test_dr = X_test_used

    # compute per-feature std
    var_array = np.sqrt(np.var(X_test_dr, axis=0))
    var_list = list(var_array)

    # avg_grad_calc should accept numpy array inputs; prefer X_test_dr
    gradient_comp = avg_grad_calc(input_var, target_var, test_prediction, X_test_dr, y_test)
    gradient_list = list(gradient_comp)

    # Sanity check: lengths must match
    if len(var_list) != len(gradient_list):
        raise ValueError("Length mismatch: var_list {} vs gradient_list {}".format(len(var_list), len(gradient_list)))

    return list(zip(var_list, gradient_list))


def main():
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='Run gradient vs variance scatter')
    parser.add_argument('--rds', nargs='+', type=int, default=[784, 331], help='list of reduced dimensions')
    args = parser.parse_args()

    # Create model_dict from arguments
    model_dict = model_dict_create()

    rd_list = args.rds

    # Load dataset specified in model_dict
    logging.info('Loading data...')
    dataset = model_dict.get('dataset', 'MNIST')

    dl = load_dataset(model_dict)
    # Normalise return shapes
    if isinstance(dl, tuple) or isinstance(dl, list):
        if len(dl) == 6:
            X_train, y_train, X_val, y_val, X_test, y_test = dl
        elif len(dl) == 4:
            X_train, y_train, X_test, y_test = dl
            X_val, y_val = None, None
        else:
            raise ValueError("load_dataset returned unexpected number of values: {}".format(len(dl)))
    else:
        raise ValueError("load_dataset returned unexpected type: {}".format(type(dl)))

    gradient_var_list = []

    for rd in rd_list:
        gradient_var_list.append(gradient_calc(rd, model_dict, X_train, y_train, X_test, y_test, X_val, y_val))

    # Ensure mag_var_scatter expects a list-of-lists and number of dims
    mag_var_scatter(gradient_var_list, len(rd_list))


if __name__ == "__main__":
    main()
