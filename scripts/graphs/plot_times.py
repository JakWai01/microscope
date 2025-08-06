import orjson
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open("../../assets/benchmark/0003_output_ocular_k2.json.json", 'rb') as fd:
    data = orjson.loads(fd.read())

benchmarks = data["benchmarks"]
preview_times = {}
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
    preview_times[benchmark["name"]] = benchmark["stats"]["mean"]


with open("../../assets/benchmark/0002_output_qiskit.json.json", 'rb') as fd:
    data = orjson.loads(fd.read())


benchmarks = data["benchmarks"]
release_times = {}
for benchmark in benchmarks:
    if benchmark["name"] in skip_names:
        continue
    release_times[benchmark["name"]] = benchmark["stats"]["mean"]

    _sanity_check = benchmark["extra_info"]["input_num_qubits"]
    if benchmark["stats"]["mean"] < preview_times[benchmark["name"]]:
        print(f"Benchmark: {benchmark['name']} is slower")
        print(benchmark)
        count += 1

names = list(preview_times)
preview_data = [preview_times[name] for name in names]
release_data = [release_times[name] for name in names]

print(len(names))
print(f"There are {count} outliers")
ratios = [release_data[i] / preview_data[i] for i in range(len(preview_data))]
df = pd.DataFrame({"benchmark": names, "Qiskit 2.1.0rc1": preview_data, "Qiskit 2.0.2": release_data, "Runtime ratio (2.0.2 / 2.1.0rc1)": ratios})
#print(df)
sns.set()
fig, ax1 = plt.subplots()
cmap = sns.color_palette("crest", as_cmap=True)

c = [qubit_counts[x] for x in names]

norm = mpl.colors.Normalize(vmin=min(c), vmax=max(c))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
ax1.scatter(preview_data, release_data, c=c, cmap=cmap, s=10)
line = np.linspace(min(release_data), max(release_data))
ax1.plot(line, line, linewidth=0.5, color='red')
ax1.set_ylabel(f"Qiskit 2.0.2 Runtime (sec.)")
ax1.set_xlabel(f"Qiskit 2.1.0rc1 Runtime (sec.)")
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_title(f"Qiskit 2.0.2 vs 2.1.0rc1 running benchpress transpile benchmarks")
plt.colorbar(sm, ax=ax1, label='Number of Qubits')
plt.tight_layout()
plt.savefig("/tmp/test_times.png", dpi=900)

fig, ax1 = plt.subplots()
ax1.semilogy(list(range(len(names))), ratios)
ax1.set_ylabel("Runtime ratio 2.0.2 / 2.1.0rc1")
ax1.set_title("Qiskit 2.0.2 vs 2.1.0rc1 runtime ratio for benchpress transpile benchmarks")
plt.tight_layout()
plt.savefig("/tmp/ratio_times.png")
import statistics
print(statistics.geometric_mean(ratios))
