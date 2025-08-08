import orjson
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("../../assets/benchmark/0002_output_qiskit.json.json", "rb") as f:
    data = orjson.loads(f.read())

records = []

for bench in data.get("benchmarks", []):
    name = bench["name"]
    params = bench["params"]
    topo = params["circ_and_topo"][1]
    info = bench["extra_info"]
    stats = bench["stats"]

    records.append(
        {
            "name": name,
            "topology": topo,
            "num_qubits": info["input_num_qubits"],
            "swap_count": info["output_circuit_operations"].get("swap", 0),
            "runtime": stats["total"],
        }
    )

df = pd.DataFrame(records).dropna()
df = df[df["swap_count"] > 0]

topologies = ["square", "heavy-hex", "linear"]
markers = ["o", "s", "D"]

sns.set_theme()
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
cmap = sns.color_palette("crest", n_colors=3)

for i, (topo, cmap, marker) in enumerate(zip(topologies, cmap, markers)):
    df_topo = df[df["topology"] == topo]

    ax = axs[i]
    ax.scatter(
        df_topo["num_qubits"],
        df_topo["swap_count"],
        s=60,
        alpha=0.8,
        color=cmap,
        edgecolor="black",
        linewidth=0.5,
        marker=marker,
        label=topo,
    )

    ax.set_yscale("log")
    ax.set_ylabel("SWAP Count", fontsize=14)
    ax.set_title(f"{topo.capitalize()} Topology", fontsize=16, weight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=12)

axs[-1].set_xlabel("Qubit Count", fontsize=14)
plt.tight_layout()
plt.savefig("/tmp/bench_circuits.png")
