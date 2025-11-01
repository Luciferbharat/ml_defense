import os
import sys
import numpy as np
import subprocess

from lib.utils.svm_utils import (
    svm_model_dict_create,
    model_creator,
    model_tester,
    model_transform,
    print_svm_output,
    save_svm_images,
)
from lib.utils.data_utils import load_dataset, get_data_shape
from lib.utils.dr_utils import dr_wrapper
from lib.attacks.svm_attacks import mult_cls_atk, acc_calc_all


def main(argv):
    """
    Main entry for strategic_svm.py
    Runs SVM classifier setup, attacks, retraining defenses, and strategic attacks.
    Saves all outputs and images for each step.
    """

    # Initialize model configuration
    model_dict = svm_model_dict_create()
    DR = model_dict.get("dim_red", None)
    rev_flag = model_dict.get("rev", 0)
    strat_flag = 1

    # Load dataset
    print("Loading data...")
    dataset = model_dict["dataset"]

    if dataset in ["MNIST", "GTSRB"]:
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
        img_flag = True
    elif dataset == "HAR":
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
        X_val, y_val = None, None
        img_flag = False
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    data_dict = get_data_shape(X_train, X_test)
    n_features = data_dict["no_of_features"]

    # Flatten and normalize
    X_train_flat = X_train.reshape(-1, n_features).astype(np.float32)
    X_test_flat = X_test.reshape(-1, n_features).astype(np.float32)
    mean = np.mean(X_train_flat, axis=0)
    X_train_flat -= mean
    X_test_flat -= mean

    # Create / load SVM model
    clf = model_creator(model_dict, X_train_flat, y_train)
    model_tester(model_dict, clf, X_test_flat, y_test)

    # Define attack parameters
    n_mag = 25
    dev_list = np.linspace(0.1, 2.5, n_mag)

    if dataset == "MNIST":
        rd_list = [784, 331, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    elif dataset == "HAR":
        rd_list = [561, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    else:
        rd_list = [n_features]

    # Run attacks on the baseline model
    print("Performing initial attack...")
    output_list = []
    for i, dev in enumerate(dev_list):
        X_adv, y_ini = mult_cls_atk(clf, X_test_flat, mean, dev)
        output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
        if img_flag:
            save_svm_images(model_dict, data_dict, X_test, X_adv, dev)

    fname = print_svm_output(model_dict, output_list, dev_list)

    # Retraining + strategic attack loop
    print("\n---- Retrain Defense + Strategic Attack ----")
    for rd in rd_list:
        print(f"\nReduced dimension: {rd}")
        output_list = []

        X_train_dr, _, dr_alg = dr_wrapper(
            X_train_flat, X_test_flat, DR, rd, y_train, rev=rev_flag
        )

        # Train new model on reduced data
        clf = model_creator(model_dict, X_train_dr, y_train, rd, rev_flag)
        clf = model_transform(model_dict, clf, dr_alg)
        model_tester(model_dict, clf, X_test_flat, y_test, rd, rev_flag)

        # Strategic adversarial attack
        print("Running strategic attack...")
        for dev in dev_list:
            X_adv, y_ini = mult_cls_atk(clf, X_test_flat, mean, dev)
            output_list.append(acc_calc_all(clf, X_adv, y_test, y_ini))
            if img_flag:
                save_svm_images(model_dict, data_dict, X_test_flat, X_adv, dev, rd, dr_alg, rev_flag)

        print_svm_output(model_dict, output_list, dev_list, rd, strat_flag, rev_flag)

    # Generate plot using gnuplot if available
    try:
        subprocess.call(
            ["gnuplot", "-e", f"mname='{fname}'", "gnu_in_loop.plg"], shell=True
        )
    except Exception as e:
        print(f"[Warning] Gnuplot not available: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
