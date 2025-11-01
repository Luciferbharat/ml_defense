import os
import numpy as np
import subprocess
from lib.utils.svm_utils import (
    svm_model_dict_create,
    model_creator,
    model_tester,
    model_transform,
    get_svm_model_name,
    resolve_path_o,
)
from lib.utils.data_utils import load_dataset, get_data_shape
from lib.utils.dr_utils import dr_wrapper
from lib.utils.plot_utils import mag_var_scatter


def main():
    """Main entry for SVM theory experiment."""
    model_dict = svm_model_dict_create()
    DR = model_dict.get("dim_red")
    rev_flag = model_dict.get("rev", 0)

    print("📦 Loading data...")
    dataset = model_dict["dataset"]

    if dataset in ["MNIST", "GTSRB"]:
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(model_dict)
    elif dataset == "HAR":
        X_train, y_train, X_test, y_test = load_dataset(model_dict)
        X_val, y_val = None, None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    data_dict = get_data_shape(X_train, X_test)
    n_features = data_dict["no_of_features"]

    # Flatten and center data
    X_train_flat = X_train.reshape(-1, n_features).astype(np.float32)
    X_test_flat = X_test.reshape(-1, n_features).astype(np.float32)
    mean = np.mean(X_train_flat, axis=0)
    X_train_flat -= mean
    X_test_flat -= mean

    # Train and evaluate base SVM
    clf = model_creator(model_dict, X_train_flat, y_train)
    model_tester(model_dict, clf, X_test_flat, y_test)

    # Setup output logging
    abs_path_o = os.path.join(resolve_path_o(model_dict), "other/")
    os.makedirs(abs_path_o, exist_ok=True)
    fname = f"norms_{get_svm_model_name(model_dict)}"
    out_path = os.path.join(abs_path_o, f"{fname}.txt")

    with open(out_path, "a", encoding="utf-8") as ofile:
        ofile.write(f"No_{DR}\n")
        for i in range(model_dict["classes"]):
            norm_val = np.linalg.norm(clf.coef_[i])
            ofile.write(f"{i},{norm_val}\n")
        ofile.write("\n\n")

        # Analyze variance vs coefficients
        var_array = np.sqrt(np.var(X_test_flat, axis=0))
        var_list = list(var_array)
        coef_var_list = []

        coef_list = list(np.abs(clf.coef_[0, :]))
        coef_var_list.append(list(zip(var_list, coef_list)))

        # Define reduced dimensions
        if dataset == "MNIST":
            rd_list = [331, 100, 80, 60, 40, 20]
        elif dataset == "HAR":
            rd_list = [561, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        else:
            rd_list = [n_features]

        # Run through reduced dimensions
        for rd in rd_list:
            print(f"🔹 Reduced dimension: {rd}")
            X_train_dr, _, dr_alg = dr_wrapper(
                X_train_flat, X_test_flat, DR, rd, y_train, rev=rev_flag
            )

            clf = model_creator(model_dict, X_train_dr, y_train, rd, rev_flag)
            clf = model_transform(model_dict, clf, dr_alg)
            model_tester(model_dict, clf, X_test_flat, y_test, rd, rev_flag)

            ofile.write(f"{DR}_{rd}\n")
            for i in range(model_dict["classes"]):
                norm_val = np.linalg.norm(clf.coef_[i])
                ofile.write(f"{i},{norm_val}\n")
            ofile.write("\n\n")

            coef_list_dr = list(np.abs(clf.coef_[0, :]))
            coef_var_list.append(list(zip(var_list, coef_list_dr)))

    # Plot magnitude-variance scatter
    print("📊 Generating coefficient-variance scatter plots...")
    mag_var_scatter(model_dict, coef_var_list, len(rd_list) + 1, rd, rev_flag)

    print(f"✅ Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
