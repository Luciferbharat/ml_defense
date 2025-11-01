import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import scipy.optimize

from ..utils.attack_utils import acc_calc_all, iterate_minibatches, grad_fn, local_fns
from ..utils.data_utils import load_dataset
from ..utils.dr_utils import gradient_transform


# --------------------------------------------------------------------------- #
# Fast Gradient Sign Method (FGSM)
# --------------------------------------------------------------------------- #
def fgs(model_dict, x_curr, y_curr, adv_x, dev_mag, b_c, gradient, dr_alg, rd):
    """
    Performs Fast Sign Gradient (FGSM) attack and stores perturbed examples in adv_x.
    """
    batch_len = x_curr.shape[0]
    delta_x = gradient(x_curr, y_curr)
    no_of_features = x_curr.shape[1]

    if dr_alg is not None:
        A = gradient_transform(model_dict, dr_alg)
        delta_x = np.dot(delta_x.reshape(batch_len, no_of_features), A)

    delta_x_sign = np.sign(delta_x)
    rev = model_dict.get("rev", None)

    if rd is None or rev is not None:
        adv_x[b_c * batch_len:(b_c + 1) * batch_len] = np.clip(
            x_curr + dev_mag * delta_x_sign, 0, 1
        )
    else:
        adv_x[b_c * batch_len:(b_c + 1) * batch_len] = x_curr + dev_mag * delta_x_sign


# --------------------------------------------------------------------------- #
# Fast Gradient (normalized gradient attack)
# --------------------------------------------------------------------------- #
def fg(model_dict, data_dict, x_curr, y_curr, x_curr_orig, adv_x, dev_mag, b_c,
       gradient, dr_alg, rd, mean):
    """
    Performs Fast Gradient attack and stores perturbed examples in adv_x.
    """
    batch_len = x_curr.shape[0]
    features_per_c = data_dict["features_per_c"]
    no_of_features = data_dict["no_of_features"]
    channels = data_dict["channels"]
    no_of_dim = data_dict["no_of_dim"]
    rev = model_dict.get("rev", None)

    delta_x = gradient(x_curr, y_curr)

    if dr_alg is not None:
        A = gradient_transform(model_dict, dr_alg)
        delta_x = np.dot(delta_x.reshape(batch_len, no_of_features), A)

    # 2D input
    if no_of_dim == 2:
        delta_x_norm = np.linalg.norm(delta_x, axis=1)
        for i in range(batch_len):
            if delta_x_norm[i] == 0.0:
                adv_x[b_c * batch_len + i] = x_curr_orig[i]
            else:
                x_adv_curr = np.clip(
                    x_curr_orig[i] + dev_mag * (delta_x[i] / delta_x_norm[i]) + mean,
                    0, 1
                )
                x_adv_curr -= mean
                if dr_alg is not None:
                    adv_x[b_c * batch_len + i] = np.dot(x_adv_curr, A.T)
                else:
                    adv_x[b_c * batch_len + i] = x_adv_curr

    # 3D input
    elif no_of_dim == 3:
        if dr_alg is not None:
            curr_dim = delta_x.shape[1]
            delta_x_norm = np.linalg.norm(
                delta_x.reshape(batch_len, channels, curr_dim), axis=2
            )
        else:
            delta_x_norm = np.linalg.norm(
                delta_x.reshape(batch_len, channels, features_per_c), axis=2
            )

        mean = mean.reshape(channels, -1)
        for i in range(batch_len):
            for j in range(channels):
                if delta_x_norm[i, j] == 0.0:
                    adv_x[b_c * batch_len + i, j] = x_curr_orig[i, j]
                else:
                    x_adv_curr = np.clip(
                        x_curr_orig[i, j]
                        + dev_mag * (delta_x[i, j] / delta_x_norm[i, j])
                        + mean[j],
                        0, 1,
                    )
                    x_adv_curr -= mean[j]
                    if dr_alg is not None:
                        x_adv_curr = np.dot(x_adv_curr.reshape(1, -1), A.T)
                    adv_x[b_c * batch_len + i, j] = x_adv_curr

    # 4D input (images)
    elif no_of_dim == 4:
        height, width = data_dict["height"], data_dict["width"]
        if dr_alg is not None:
            curr_dim = delta_x.shape[1]
            delta_x_norm = np.linalg.norm(
                delta_x.reshape(batch_len, channels, curr_dim), axis=2
            )
        else:
            delta_x_norm = np.linalg.norm(
                delta_x.reshape(batch_len, channels, features_per_c), axis=2
            )
        for i in range(batch_len):
            for j in range(channels):
                if delta_x_norm[i, j] == 0.0:
                    adv_x[b_c * batch_len + i, j] = x_curr_orig[i, j]
                else:
                    x_adv_curr = np.clip(
                        x_curr_orig[i, j]
                        + dev_mag * (delta_x[i, j] / delta_x_norm[i, j])
                        + mean[j],
                        0, 1,
                    )
                    x_adv_curr -= mean[j]
                    if dr_alg is not None:
                        x_adv_curr = np.dot(x_adv_curr.reshape(1, -1), A.T)
                    adv_x[b_c * batch_len + i, j] = x_adv_curr.reshape(
                        1, channels, height, width
                    )


# --------------------------------------------------------------------------- #
# Attack wrapper
# --------------------------------------------------------------------------- #
def attack_wrapper(model_dict, data_dict, model, dev_list, X_test, y_test, mean,
                   dr_alg=None, rd=None):
    """
    Creates adversarial examples using FGSM or FG attacks.
    """
    adv_len = data_dict["test_len"]
    no_of_features = data_dict["no_of_features"]
    no_of_dim = data_dict["no_of_dim"]
    channels = data_dict["channels"]
    dataset = model_dict["dataset"]

    n_mags = len(dev_list)
    adv_x_all = np.zeros((adv_len, no_of_features, n_mags))

    # Gradient computation (using torch autograd)
    def gradient_fn(x_batch, y_batch):
        x_batch_t = torch.tensor(x_batch, dtype=torch.float32, requires_grad=True)
        y_batch_t = torch.tensor(y_batch, dtype=torch.long)
        outputs = model(x_batch_t)
        loss = F.cross_entropy(outputs, y_batch_t)
        loss.backward()
        grad = x_batch_t.grad.detach().numpy()
        return grad

    # Prepare arrays
    if no_of_dim == 2:
        adv_x = np.zeros((adv_len, no_of_features))
    elif no_of_dim == 3:
        features = data_dict["features_per_c"]
        adv_x = np.zeros((adv_len, channels, features))
    elif no_of_dim == 4:
        height, width = data_dict["height"], data_dict["width"]
        adv_x = np.zeros((adv_len, channels, height, width))

    if dataset in ["MNIST", "GTSRB"]:
        _, _, _, _, X_test_orig, _ = load_dataset(model_dict)
    elif dataset == "HAR":
        _, _, X_test_orig, _ = load_dataset(model_dict)
    else:
        X_test_orig = X_test.copy()

    X_test_orig -= mean

    o_list = []
    mag_count = 0
    for dev_mag in dev_list:
        adv_x.fill(0)
        start_time = time.time()
        batch_len = 1000
        b_c = 0
        for x_curr, y_curr in iterate_minibatches(X_test, y_test, batch_len):
            x_curr_orig = X_test_orig[b_c * batch_len:(b_c + 1) * batch_len]
            if model_dict["attack"] == "fgs":
                fgs(model_dict, x_curr, y_curr, adv_x, dev_mag, b_c,
                    gradient_fn, dr_alg, rd)
            elif model_dict["attack"] == "fg":
                fg(model_dict, data_dict, x_curr, y_curr, x_curr_orig, adv_x,
                   dev_mag, b_c, gradient_fn, dr_alg, rd, mean)
            b_c += 1

        adv_x_all[:, :, mag_count] = adv_x.reshape((adv_len, no_of_features))
        o_list.append({"magnitude": dev_mag})
        print(f"⚡ FGSM done (ε={dev_mag:.3f}) — took {time.time() - start_time:.2f}s")
        mag_count += 1

    return adv_x_all, o_list


# --------------------------------------------------------------------------- #
# L-BFGS Attack
# --------------------------------------------------------------------------- #
def l_bfgs_attack(model, X_test, y_test, rd=None, max_dev=None):
    """
    Performs L-BFGS adversarial attack on a subset of test data.
    """
    C = 0.7
    trial_size = min(1000, len(X_test))
    X_test, y_test = X_test[:trial_size], y_test[:trial_size]

    adv_x, deviation_list = [], []
    count_wrong = count_correct = 0.0
    deviation = magnitude = 0.0

    for i in range(trial_size):
        print(f"▶ Iter {i+1}/{trial_size}")
        X_curr = X_test[i:i+1]
        y_curr = np.random.choice(y_test[y_test != y_test[i]], size=1)

        ini_pred = model(torch.tensor(X_curr, dtype=torch.float32)).argmax(1).item()

        def f(x):
            x_t = torch.tensor(X_curr + x.reshape(1, -1), dtype=torch.float32)
            loss = F.cross_entropy(model(x_t), torch.tensor(y_curr))
            return C * np.linalg.norm(x) + loss.item()

        x_0 = np.zeros(rd)
        r, fval, info = scipy.optimize.fmin_l_bfgs_b(f, x_0, approx_grad=True)
        adv_sample = X_curr + r.reshape(1, -1)
        adv_x.append(adv_sample)
        deviation_list.append(np.linalg.norm(r))

        pred_new = model(torch.tensor(adv_sample, dtype=torch.float32)).argmax(1).item()
        if ini_pred == y_test[i]:
            count_correct += 1
            magnitude += np.linalg.norm(X_curr)
        if pred_new != ini_pred and ini_pred == y_test[i]:
            if max_dev is None or np.linalg.norm(r) < max_dev:
                count_wrong += 1
            deviation += np.linalg.norm(r)

    adv_x = np.array(adv_x)
    results = [
        deviation / max(count_wrong, 1),
        magnitude / max(count_correct, 1),
        (count_wrong / max(count_correct, 1)) * 100,
    ]
    print(f"📈 Avg deviation: {results[0]:.4f}, Avg magnitude: {results[1]:.4f}, "
          f"Success rate: {results[2]:.2f}%")

    return adv_x, results, deviation_list
