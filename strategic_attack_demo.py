#!/usr/bin/env python3
"""
strategic_attack_demo.py (modernized)

Performs strategic adversarial attacks on NN models using multiple
dimensionality reduction levels. Compatible with Theano-PyMC + Lasagne.
"""

import os
import sys
import numpy as np
import theano
import theano.tensor as T
import multiprocessing
from functools import partial
from matplotlib import pyplot as plt

# Local imports
from lib.utils.data_utils import load_dataset
from lib.utils.model_utils import model_dict_create, model_setup, print_output
from lib.attacks.nn_attacks import attack_wrapper
# from lib.utils.model_utils import save_images  # Uncomment if needed


# ----------------------------------------------------------------------------- #
def strategic_attack(
    rd: int,
    model_dict: dict,
    dev_list: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mean: np.ndarray,
    X_val=None,
    y_val=None,
) -> None:
    """
    Run a strategic adversarial attack for a given reduced dimension (rd).
    """

    print(f"[INFO] Running strategic attack for rd = {rd}")

    # Model setup for this dimension
    rev_flag = model_dict.get("rev", False)
    layer_flag = None
    dim_red = model_dict.get("dim_red")

    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = model_setup(
        model_dict, X_train, y_train, X_test, y_test, X_val, y_val, rd, layer=layer_flag
    )

    # Generate adversarial examples
    adv_x_all, output_list = attack_wrapper(
        model_dict,
        data_dict,
        input_var,
        target_var,
        test_prediction,
        dev_list,
        X_test,
        y_test,
        mean,
        dr_alg,
        rd,
    )

    # Print results to file
    print_output(model_dict, output_list, dev_list, is_defense=False, rd=rd, strat_flag=1)

    # Optionally save adversarial examples as images
    # if dim_red in ('pca', 'dca', None):
    #     save_images(model_dict, data_dict, X_test, adv_x_all, dev_list, rd, dr_alg, rev=rev_flag)

    print(f"[SUCCESS] Completed strategic attack for rd = {rd}")


# ----------------------------------------------------------------------------- #
def main() -> None:
    """
    Entry point for strategic_attack_demo.py.
    Loads dataset, initializes parameters, and runs attacks for all rd values.
    """

    # Initialize model dictionary
    model_dict = model_dict_create()
    dataset = model_dict.get("dataset")

    # Deviation magnitudes
    no_of_mags = 50
    dev_list = np.linspace(0.1, 5.0, no_of_mags)

    print(f"[INFO] Loading dataset: {dataset}")
    if dataset in ("MNIST", "GTSRB"):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
        rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    elif dataset == "HAR":
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
        X_val = y_val = None
        rd_list = [561, 300, 150, 100, 50, 25, 10]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Normalize by mean subtraction
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_test -= mean
    if X_val is not None:
        X_val -= mean

    print("[INFO] Creating initial adversarial samples...")
    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = model_setup(
        model_dict, X_train, y_train, X_test, y_test, X_val, y_val
    )

    adv_x_ini, output_list = attack_wrapper(
        model_dict, data_dict, input_var, target_var, test_prediction,
        dev_list, X_test, y_test, mean
    )

    print_output(model_dict, output_list, dev_list)
    # save_images(model_dict, data_dict, X_test, adv_x_ini, dev_list)  # Uncomment if needed

    # --- Run strategic attacks for all reduced dimensions ---
    print(f"[INFO] Running strategic attacks for {len(rd_list)} dimensions...")

    # If you want to parallelize:
    # pool = multiprocessing.Pool(processes=min(8, len(rd_list)))
    # func = partial(
    #     strategic_attack,
    #     model_dict=model_dict,
    #     dev_list=dev_list,
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     mean=mean,
    #     X_val=X_val,
    #     y_val=y_val,
    # )
    # pool.map(func, rd_list)
    # pool.close()
    # pool.join()

    # Sequential version (simpler debugging)
    for rd in rd_list:
        strategic_attack(rd, model_dict, dev_list, X_train, y_train, X_test, y_test, mean, X_val, y_val)

    print("[SUCCESS] All strategic attacks completed.")


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
