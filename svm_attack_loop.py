import subprocess
import multiprocessing


def run_attack_for_C(C_value: str):
    """Run strategic SVM attacks for a given regularization parameter C."""
    for whiten in [1, 2, 3]:
        print(f"🚀 Running attack for C={C_value}, antiwhiten={whiten}")
        cmd = f"python strategic_svm.py -dr antiwhiten{whiten} -C {C_value}"
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[Error] Attack failed for C={C_value}, whiten={whiten}: {e}")


if __name__ == "__main__":
    C_list = ["1e-05", "1e-04", "1e-03", "1e-02", "1e-01", "1e+00", "1e+01"]

    print("🔁 Starting parallel attack runs...")
    with multiprocessing.Pool(processes=5) as pool:
        pool.map(run_attack_for_C, C_list)
    print("✅ All attacks completed successfully.")
