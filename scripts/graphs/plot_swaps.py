import orjson
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


with open("/home/jakob/Documents/Projects/microscope/assets/benchmark/0003_output_ocular_k2.json.json", 'rb') as fd:
    data = orjson.loads(fd.read())

benchmarks = data["benchmarks"]
preview_swaps = {}
qubit_counts = {}
skip_names = set()
count = 0
for benchmark in benchmarks:
    try:
        qubit_counts[benchmark["name"]] = benchmark["extra_info"]["input_num_qubits"]
    except KeyError:
        print(benchmark["name"])
        skip_names.add(benchmark["name"])
        continue
    try:
        if benchmark["params"]["circ_and_topo"][1] == "square":
            preview_swaps[benchmark["name"]] = benchmark["extra_info"]["output_circuit_operations"]["swap"]
    except KeyError:
        skip_names.add(benchmark["name"])
        continue


with open("/home/jakob/Documents/Projects/microscope/assets/benchmark/0002_output_qiskit.json.json", 'rb') as fd: 
    data = orjson.loads(fd.read())


benchmarks = data["benchmarks"]
release_swaps = {}
for benchmark in benchmarks:
    if benchmark["name"] in skip_names:
        continue

    if benchmark["params"]["circ_and_topo"][1] == "square":
        release_swaps[benchmark["name"]] = benchmark["extra_info"]["output_circuit_operations"]["swap"]
    else:
        continue

    _sanity_check = benchmark["extra_info"]["input_num_qubits"]
    if release_swaps[benchmark["name"]] < preview_swaps[benchmark["name"]]:
        count += 1

names = list(preview_swaps)
preview_data = [preview_swaps[name] for name in names]
release_data = [release_swaps[name] for name in names]

print(len(skip_names))
print(len(names))
print(f"There are {count} release results better than the preview")
print(f"There are {len(names) - count} preview results equal or better than the release")
ratios = []
for i in range(len(preview_data)):
    if preview_data[i]:
        ratios.append(release_data[i] / preview_data[i])
    else:
        ratios.append(release_data[i])
#ratios = [release_data[i] / preview_data[i] for i in range(len(preview_data))]
df = pd.DataFrame({"benchmark": names, "k=2": preview_data, "Qiskit": release_data, "swap ratio (k=2 / qiskit)": ratios})
#print(df)
sns.set()
fig, ax1 = plt.subplots()
cmap = sns.color_palette("crest", as_cmap=True)

c = [qubit_counts[x] for x in names]

norm = mpl.colors.Normalize(vmin=min(c), vmax=max(c))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
ax1.scatter(preview_data, release_data, c=c, cmap=cmap, s=5)
line = np.linspace(min(release_data), max(release_data))
ax1.plot(line, line, linewidth=1, color='black', linestyle="dashed")
ax1.set_ylabel(f"Qiskit swaps")
ax1.set_xlabel(f"k=2 swaps")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_title(f"Qiskit vs k=2 running benchpress transpile benchmarks")
plt.colorbar(sm, ax=ax1, label='Number of Qubits')
plt.tight_layout()
plt.savefig("/tmp/test_swaps.png", dpi=900)

fig, ax1 = plt.subplots()
ax1.semilogy(list(range(len(names))), ratios)
ax1.set_ylabel("swap ratio k=2 / Qiskit")
ax1.set_title("k=2 vs Qiskit swap count ratio for benchpress transpile benchmarks")
plt.tight_layout()
plt.savefig("/tmp/ratio_swaps.png")
import statistics
print(statistics.geometric_mean(ratios))
