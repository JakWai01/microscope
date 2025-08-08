import orjson
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

topologies = ["heavy-hex", "square", "linear"]

with open("/home/jakob/Documents/Projects/microscope/assets/benchmark/0003_output_ocular_k2.json.json", 'rb') as fd:
    data_preview = orjson.loads(fd.read())

with open("/home/jakob/Documents/Projects/microscope/assets/benchmark/0002_output_qiskit.json.json", 'rb') as fd: 
    data_release = orjson.loads(fd.read())

benchmarks_preview = data_preview["benchmarks"]
benchmarks_release = data_release["benchmarks"]

sns.set()

fig = plt.figure(figsize=(20, 6))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.07], wspace=0.3)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

cmap = sns.color_palette("crest", as_cmap=True)

all_qubit_counts = {}
for benchmark in benchmarks_preview:
    try:
        all_qubit_counts[benchmark["name"]] = benchmark["extra_info"]["input_num_qubits"]
    except KeyError:
        continue

norm = mpl.colors.Normalize(vmin=min(all_qubit_counts.values()), vmax=max(all_qubit_counts.values()))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for ax, topo in zip(axs, topologies):
    preview_swaps = {}
    release_swaps = {}
    skip_names = set()

    for benchmark in benchmarks_preview:
        try:
            if benchmark["params"]["circ_and_topo"][1] == topo:
                preview_swaps[benchmark["name"]] = benchmark["extra_info"]["output_circuit_operations"]["swap"]
        except KeyError:
            skip_names.add(benchmark["name"])

    for benchmark in benchmarks_release:
        if benchmark["name"] in skip_names:
            continue
        if benchmark["params"]["circ_and_topo"][1] == topo:
            release_swaps[benchmark["name"]] = benchmark["extra_info"]["output_circuit_operations"]["swap"]
        else:
            continue

    names = list(preview_swaps)
    preview_data = [preview_swaps[name] for name in names if name in release_swaps]
    release_data = [release_swaps[name] for name in names if name in release_swaps]
    names = [name for name in names if name in release_swaps]

    c = [all_qubit_counts[x] for x in names]

    ax.scatter(preview_data, release_data, c=c, cmap=cmap, s=20)

    all_data = preview_data + release_data
    line_min = max(min(all_data), 1)  # avoid zero or negative for log scale
    line_max = max(all_data)
    line = np.linspace(line_min, line_max, 100)
    ax.plot(line, line, linewidth=1, color='black', linestyle="dashed")

    ax.set_title(f"Topology: {topo}", fontweight='bold', fontsize=14)
    ax.set_xlabel("k=2 swaps", fontsize=12)
    ax.set_ylabel("Qiskit swaps", fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # automatic axis limits
    ax.set_xlim(1, 1e5)
    ax.set_ylim(1, 1e5)

cbar_ax = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Number of Qubits')

plt.suptitle("Qiskit vs k=2 swaps across topologies", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("/tmp/combined_topologies_swaps.png", dpi=900)
plt.show()
