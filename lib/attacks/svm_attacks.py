"""Utility file containing attack algorithms for SVM."""

import numpy as np
from typing import Tuple

# ------------------------------------------------------------------------------ #
def min_dist_calc(x: np.ndarray, clf) -> Tuple[int, float]:
    """
    Find the class whose hyperplane is closest to the given sample x
    and calculate that minimum distance.

    Args:
        x (np.ndarray): Input sample (1D vector)
        clf: Trained linear SVM classifier (with coef_ and intercept_ attributes)

    Returns:
        Tuple[int, float]: (closest class index, minimum distance)
    """
    x_ini = x.reshape(1, -1)
    ini_class = int(clf.predict(x_ini)[0])
    w_ini = clf.coef_[ini_class, :]

    distances = clf.decision_function(x_ini)
    num_classes = clf.intercept_.shape[0]

    d_list = []
    i_list = []

    for j in range(num_classes):
        if j == ini_class:
            continue
        w_curr = clf.coef_[j, :]
        # Distance to hyperplane difference
        dist = abs(distances[0, j] - distances[0, ini_class]) / np.linalg.norm(w_curr - w_ini)
        d_list.append(dist)
        i_list.append(j)

    # Find class with minimum distance
    min_index = i_list[int(np.argmin(d_list))]
    min_dist = min(d_list)

    return min_index, min_dist
# ------------------------------------------------------------------------------ #


def mult_cls_atk(
    clf,
    X_test: np.ndarray,
    mean: np.ndarray,
    dev_mag: float,
    img_flag: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate adversarial samples for a multi-class linear SVM.

    Args:
        clf: Trained linear SVM classifier
        X_test (np.ndarray): Test set samples
        mean (np.ndarray): Mean of training data (for normalization)
        dev_mag (float): Perturbation magnitude
        img_flag (bool): Whether input represents image data (controls clipping)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X_adv: Adversarial samples
            - y_ini: Original predicted labels
    """

    test_len = len(X_test)
    X_adv = np.zeros_like(X_test)
    y_ini = np.zeros(test_len, dtype=int)

    for i in range(test_len):
        x_ini = X_test[i, :].reshape(1, -1)
        ini_class = int(clf.predict(x_ini)[0])
        min_index, _ = min_dist_calc(x_ini, clf)

        w_ini = clf.coef_[ini_class, :]
        w_min = clf.coef_[min_index, :]

        # Create adversarial perturbation in direction between hyperplanes
        direction = (w_ini - w_min) / np.linalg.norm(w_ini - w_min)
        x_adv = x_ini - dev_mag * direction

        X_adv[i, :] = x_adv
        y_ini[i] = ini_class

    # Clip values if input data represents images
    if img_flag:
        X_adv += mean
        np.clip(X_adv, 0, 1, out=X_adv)
        X_adv -= mean

    return X_adv, y_ini
# ------------------------------------------------------------------------------ #
