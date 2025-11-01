import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# ----------------- CONFIG -----------------
FONT = {'size': 17}
OUTPUT_DATA_DIR = "output_data"
PLOT_OUTPUT_DIR = "plots"
INPUT_FILE = "FSG_mod_MNIST_nn_2_100_strategic.txt"
LIST_DIM = [784, 331, 100, 50, 40, 30, 20, 10]
TITLE = "Re-training defense for MNIST data against FSG attack\nModel: FC100-100-10"
X_LABEL = "Adversarial perturbation"
Y_LABEL = "Adversarial success"
SAVE_FILE = "MNIST_nn_2_strategic.png"
# ------------------------------------------

matplotlib.rc('font', **FONT)
COLORS = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

# Dynamically pick markers
MARKERS = [
    m for m in Line2D.markers
    if isinstance(m, str) and len(m) == 1 and m != ' '
]

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
abs_path_o = os.path.join(script_dir, OUTPUT_DATA_DIR)
abs_path_p = os.path.join(script_dir, PLOT_OUTPUT_DIR)

if not os.path.exists(abs_path_p):
    os.makedirs(abs_path_p)

# ----------------- PLOTTING -----------------
fig, ax = plt.subplots(figsize=(12, 9))
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

handle_list = []
dims = len(LIST_DIM)

data_path = os.path.join(abs_path_o, INPUT_FILE)
if not os.path.isfile(data_path):
    raise FileNotFoundError(f"Input file not found: {data_path}")

print(f"[INFO] Reading data from: {data_path}")

for idx, dim in enumerate(LIST_DIM, start=1):
    print(f"[DEBUG] Processing dimension: {dim}")
    color = COLORS[idx % len(COLORS)]
    marker = MARKERS[idx % len(MARKERS)]
    try:
        curr_array = np.genfromtxt(
            data_path,
            delimiter=',',
            skip_header=2 + 52 * (idx - 1),
            skip_footer=52 * (dims - idx)
        )
    except Exception as e:
        print(f"[ERROR] Failed to read section for dim={dim}: {e}")
        continue

    if curr_array.ndim < 2 or curr_array.shape[1] <= 5:
        print(f"[WARN] Skipping dim={dim} (invalid data format)")
        continue

    handle = ax.plot(
        curr_array[:, 0],
        curr_array[:, 5],
        linestyle='-',
        marker=marker,
        color=color,
        markersize=8,
        label=str(dim)
    )
    handle_list.append(handle)

ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)
ax.set_title(TITLE)
ax.legend(loc='upper left', fontsize=14)
plt.tight_layout()

save_path = os.path.join(abs_path_p, SAVE_FILE)
plt.savefig(save_path, bbox_inches='tight')
plt.close(fig)

print(f"[SUCCESS] Plot saved to: {save_path}")
