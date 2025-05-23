import matplotlib.pyplot as plt

fig, ax = plt.subplots()

es = [5, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
swaps = [
    21710,
    18076,
    16252,
    16356,
    16400,
    16400,
    16505,
    16558,
    16881,
    18000,
    17500,
    17700,
    17700,
]

ax.plot(es, swaps, label=f"ham_heis_graph_2D_grid_pbc_qubitnodes_Lx_5_Ly_186_h_3.qasm")

ax.legend()

ax.set(
    xlabel="Extended-Set Size",
    ylabel="Swaps",
    title="Extended-Set Size Scaling",
    xlim=(0, 8),
    xticks=range(0, 1001, 100),
)
ax.grid()

plt.xlim((0, 1000))

plt.show()
