import json
import pandas as pd
import matplotlib.pyplot as plt

# with open("../../assets/benchmark/0001_output.json.json") as f:
# with open("../../assets/benchmark/0003_output_ocular_k2.json.json") as f:
with open("../../assets/benchmark/0002_output_qiskit.json.json") as f:
    data = json.load(f)

records = []
for bench in data.get("benchmarks", []):
    name = bench.get("name", "")
    params = bench.get("params", "")
    topo = params.get("circ_and_topo")[1]
    info = bench.get("extra_info", {})
    stats = bench.get("stats", {})

    records.append({
        "name": name,
        "topology": topo,
        "num_qubits": info.get("input_num_qubits"),
        "swap_count": info.get("output_circuit_operations", 0).get("swap", 0),
        "runtime": stats.get("total", None)
    })

df = pd.DataFrame(records)

df = df.dropna()

# Filter out all values with swap_count == 0. Since we are optimizing for SWAPs,
# other circuits (like e.g. all-to-all topology) are not relvant to us
df = df[df["swap_count"] > 0]

name_width = df['name'].str.len().max()
print(df.to_string(formatters={'name': '{{:<{}}}'.format(name_width).format}))

df_linear = df[df["topology"] == "linear"].copy()
df_heavyhex = df[df["topology"] == "heavy-hex"].copy()
df_square = df[df["topology"] == "square"].copy()

topologies = ["square", "heavy-hex", "linear"]
fig, axs = plt.subplots(nrows=len(topologies), ncols=1, figsize=(14, 12), sharex=False, sharey=False)

for i, topo in enumerate(topologies):
    df_topo = df[df["topology"] == topo]

    ax = axs[i]
    x = df_topo["num_qubits"]
    y = df_topo["swap_count"]

    ax.scatter(x, y, s=40, alpha=0.7, label="swap_count")
    ax.grid(True, which="both", ls='--', lw=0.5)

    ax.set_ylabel(topo)

    ax.set_xlabel("Qubit count")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()