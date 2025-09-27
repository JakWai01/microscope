import orjson
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# User-provided file paths for each k value
# Example usage:
# python plot_swaps.py --ocular1 path1 --ocular2 path2 --ocular3 path3 --qiskit path_qiskit
import argparse

parser = argparse.ArgumentParser(description="Compare K-SWAP SABRE and Qiskit swap counts for k=1,2,3")
parser.add_argument('--ocular1', type=str, required=True, help='Path to ocular k=1 benchmark_swap_stats.json')
parser.add_argument('--ocular2', type=str, required=True, help='Path to ocular k=2 benchmark_swap_stats.json')
parser.add_argument('--ocular3', type=str, required=True, help='Path to ocular k=3 benchmark_swap_stats.json')
parser.add_argument('--qiskit', type=str, required=True, help='Path to qiskit benchmark_swap_stats.json')
parser.add_argument('--out', type=str, default='/tmp/combined_topologies_swaps_grid.png', help='Output file path')
args = parser.parse_args()

topologies = ["linear", "heavy-hex", "square"]
k_labels = ["k=1", "k=2", "k=3"]
ocular_files = [args.ocular1, args.ocular2, args.ocular3]

# Load all ocular data
ocular_data = []
for path in ocular_files:
    with open(path, "rb") as fd:
        ocular_data.append(orjson.loads(fd.read())["benchmarks"])

# Load qiskit data
with open(args.qiskit, "rb") as fd:
    data_qiskit = orjson.loads(fd.read())
qiskit = data_qiskit["benchmarks"]

sns.set_theme()
fig = plt.figure(figsize=(18, 18))
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.07], wspace=0.3, hspace=0.3)
axs = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]

cmap = sns.color_palette("crest", as_cmap=True)

# Collect all qubit counts for color normalization
all_qubit_counts = {}
for ocular in ocular_data:
    for benchmark in ocular:
        try:
            all_qubit_counts[benchmark["name"]] = benchmark["qubits"]
        except KeyError:
            continue

norm = mpl.colors.Normalize(
    vmin=min(all_qubit_counts.values()), vmax=max(all_qubit_counts.values())
)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for i, (ocular, k_label) in enumerate(zip(ocular_data, k_labels)):
    for j, topo in enumerate(topologies):
        ax = axs[i][j]
        ocular_swaps = {}
        qiskit_swaps = {}
        skip_names = set()

        for benchmark in ocular:
            try:
                if benchmark["topology"] == topo:
                    if benchmark["swap_stats"]["count"] < 10:
                        skip_names.add(benchmark["name"])
                        continue
                    ocular_swaps[benchmark["name"]] = benchmark["swap_stats"]["min"]
            except KeyError:
                skip_names.add(benchmark["name"])

        for benchmark in qiskit:
            if benchmark["name"] in skip_names:
                continue
            try:
                if benchmark["topology"] == topo:
                    qiskit_swaps[benchmark["name"]] = benchmark["swap_stats"]["min"]
            except KeyError:
                continue
            else:
                continue

        names = list(ocular_swaps)
        preview_data = [ocular_swaps[name] for name in names if name in qiskit_swaps]
        release_data = [qiskit_swaps[name] for name in names if name in qiskit_swaps]
        names = [name for name in names if name in qiskit_swaps]
        c = [all_qubit_counts[x] for x in names]

        ax.scatter(preview_data, release_data, c=c, cmap=cmap, s=20)
        all_data = preview_data + release_data
        if all_data:
            line_min = max(min(all_data), 1)
            line_max = max(all_data)
            line = np.linspace(line_min, line_max, 100)
            ax.plot(line, line, linewidth=1, color="black", linestyle="dashed")
            ax.set_xlim(1, 1e5)
            ax.set_ylim(1, 1e5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{k_label}, {topo}", fontweight="bold", fontsize=13)
        ax.set_xlabel("K-SWAP SABRE Swaps", fontsize=11)
        ax.set_ylabel("Qiskit SABRE Swaps", fontsize=11)

cbar_ax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Number of Qubits")

plt.suptitle("Qiskit SABRE vs K-SWAP SABRE (k=1,2,3) across Topologies\n(topologies sorted according to connectivity)", fontsize=16, fontweight="bold")  # default ~1.0, lower moves it closer to the figure
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig("sabre_swaps_across_topologies.pdf", 
            format="pdf", bbox_inches="tight")
plt.show()
