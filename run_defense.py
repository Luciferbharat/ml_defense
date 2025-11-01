#!/usr/bin/env python3
"""
run_defense.py (modernized)

Main script to generate adversarial samples, evaluate attacks,
and apply defenses for neural network models.
"""

import os
import sys
import time
import argparse
import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Local imports
from lib.utils.data_utils import load_dataset
from lib.utils.model_utils import model_dict_create, model_setup, print_output, save_images
from lib.attacks.nn_attacks import attack_wrapper
from lib.defenses.nn_defenses import recons_defense, retrain_defense


# ----------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run adversarial defense experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name in model_dict (optional)"
    )
    parser.add_argument(
        "--defense",
        type=str,
        choices=["retrain", "recons"],
        help="Defense type to apply"
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------- #
def main(argv=None):
    """
    Create adversarial examples and evaluate attack.
    Implement defense and re-evaluate same attack (defense-unaware).
    """
    args = parse_args() if argv is None else argv

    # Parameters
    batchsize = 500
    no_of_mags = 50
    dev_list = np.linspace(0.1, 5.0, no_of_mags)
    rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    # Model dictionary setup
    model_dict = model_dict_create()

    if args.dataset:
        model_dict["dataset"] = args.dataset
    if args.defense:
        model_dict["defense"] = args.defense

    print(f"[INFO] Using dataset: {model_dict['dataset']}")
    print(f"[INFO] Selected defense: {model_dict.get('defense', 'None')}")

    # Load dataset
    print("[INFO] Loading data...")
    dataset = model_dict["dataset"]
    if dataset in ("MNIST", "GTSRB"):
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
    elif dataset == "HAR":
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
        X_val = y_val = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Model setup
    data_dict, test_prediction, dr_alg, X_test, input_var, target_var = model_setup(
        model_dict, X_train, y_train, X_test, y_test, X_val, y_val
    )

    # Run attack
    print("[INFO] Creating adversarial samples...")
    adv_x_ini, output_list = attack_wrapper(
        model_dict, data_dict, input_var, target_var, test_prediction,
        dev_list, X_test, y_test
    )

    print_output(model_dict, output_list, dev_list)
    save_images(model_dict, data_dict, X_test, adv_x_ini, dev_list)

    # Apply defense(s)
    defense = model_dict.get("defense")
    if defense:
        print(f"[INFO] Applying defense: {defense}")
        for rd in rd_list:
            if defense == "recons":
                recons_defense(
                    model_dict, data_dict, input_var, target_var,
                    test_prediction, dev_list, adv_x_ini, rd,
                    X_train, y_train, X_test, y_test
                )
            elif defense == "retrain":
                retrain_defense(
                    model_dict, dev_list, adv_x_ini, rd,
                    X_train, y_train, X_test, y_test, X_val, y_val
                )
            else:
                print(f"[WARN] Unknown defense type: {defense}")
    else:
        print("[INFO] No defense selected. Skipping defense phase.")

    print("[SUCCESS] Defense run complete.")


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
