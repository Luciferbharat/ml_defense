"""
Neural Network Defense Methods

Implements reconstruction-based and retraining-based defenses against
adversarial examples for neural networks. Compatible with Theano and
existing project utilities.
"""

import os
import numpy as np
import theano
import theano.tensor as T
from os.path import dirname

from sklearn.decomposition import PCA

from ..utils.theano_utils import *
from ..utils.lasagne_utils import *
from ..utils.data_utils import *
from ..utils.attack_utils import *
from ..utils.model_utils import *
from ..utils.dr_utils import *

# ------------------------------------------------------------------------------ #
def recons_defense(
    model_dict: dict,
    data_dict: dict,
    input_var,
    target_var,
    test_prediction,
    dev_list: list,
    adv_x_ini: np.ndarray,
    rd: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Evaluate reconstruction-based defense effectiveness against adversarial attacks.

    Args:
        model_dict (dict): Model configuration dictionary
        data_dict (dict): Dataset metadata
        input_var: Theano symbolic input variable
        target_var: Theano symbolic output variable
        test_prediction: Model prediction tensor
        dev_list (list): List of perturbation magnitudes
        adv_x_ini (np.ndarray): Initial adversarial examples
        rd (int): Reduced dimensionality
        X_train, y_train, X_test, y_test: Dataset splits
    """

    rev_flag = model_dict.get("rev", False)
    dim_red = model_dict.get("dim_red", "pca")
    X_val = None

    print(f"Applying {dim_red} with rd={rd} over training data...")

    # Dimensionality reduction
    X_train, X_test, dr_alg = dr_wrapper(
        X_train, X_test, dim_red, rd, X_val, rev_flag
    )

    # Evaluate on reconstructed inputs
    test_model_eval(model_dict, input_var, target_var, test_prediction, X_test, y_test, rd)

    validator, indexer, predictor, confidence = local_fns(
        input_var, target_var, test_prediction
    )

    indices_c = indexer(X_test, y_test)
    i_c = np.where(indices_c == 1)[0]

    adv_len = len(adv_x_ini)
    dev_len = len(dev_list)
    no_of_features = data_dict["no_of_features"]

    adv_x = np.zeros((adv_len, no_of_features, dev_len))
    output_list = []

    for mag_idx, mag in enumerate(dev_list):
        X_adv = dr_alg.transform(adv_x_ini[:, :, mag_idx])
        X_adv_rev = invert_dr(X_adv, dr_alg, dim_red)
        adv_x[:, :, mag_idx] = X_adv_rev

        X_adv_rev = reshape_data(X_adv_rev, data_dict, rd, rev=rev_flag)

        acc_results = acc_calc_all(
            X_adv_rev, y_test, X_test, i_c, validator, indexer, predictor, confidence
        )
        output_list.append(acc_results)

    print_output(model_dict, output_list, dev_list, is_defense=True, rd=rd)

    # Save reconstructed adversarial images
    save_images(model_dict, data_dict, X_test, adv_x, dev_list, rd, dr_alg)
# ------------------------------------------------------------------------------ #


def retrain_defense(
    model_dict: dict,
    dev_list: list,
    adv_x_ini: np.ndarray,
    rd: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """
    Evaluate retraining-based defense effectiveness on adversarial examples.

    Args:
        model_dict (dict): Model configuration
        dev_list (list): List of perturbation magnitudes
        adv_x_ini (np.ndarray): Initial adversarial samples
        rd (int): Reduced dimension
        X_train, y_train, X_test, y_test, X_val, y_val: Dataset splits
    """

    # Setup model and dimensionality reduction
    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = model_setup(
        model_dict, X_train, y_train, X_test, y_test, X_val, y_val, rd
    )

    validator, indexer, predictor, confidence = local_fns(
        input_var, target_var, test_prediction
    )

    indices_c = indexer(X_test, y_test)
    i_c = np.where(indices_c == 1)[0]

    adv_len = len(adv_x_ini)
    dev_len = len(dev_list)
    adv_x = np.zeros((adv_len, rd, dev_len))

    output_list = []

    for mag_idx, mag in enumerate(dev_list):
        X_adv = dr_alg.transform(adv_x_ini[:, :, mag_idx])
        adv_x[:, :, mag_idx] = X_adv

        X_adv = reshape_data(X_adv, data_dict, rd)
        acc_results = acc_calc_all(
            X_adv, y_test, X_test, i_c, validator, indexer, predictor, confidence
        )
        output_list.append(acc_results)

    print_output(model_dict, output_list, dev_list, is_defense=True, rd=rd)

    # Optionally save adversarial samples after retraining
    # save_images(model_dict, data_dict, X_test, adv_x, dev_list, rd, dr_alg)
# ------------------------------------------------------------------------------ #
