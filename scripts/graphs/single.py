import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Set the style and color palette to match plot_swaps.py
sns.set_theme(style='whitegrid')
palette = sns.color_palette("crest", n_colors=10)
sns.set_palette(palette)

# Data from your SABRE implementation
k_values = [1, 2, 3, 4, 5, 6]

# Swap counts for each k value (10 runs each)
swaps_data = {
    1: [2018.0, 2085.0, 2092.0, 1973.0, 2053.0, 2017.0, 2025.0, 2037.0, 2037.0, 2055.0],
    2: [1990.0, 2008.0, 2022.0, 2004.0, 2016.0, 1948.0, 2000.0, 1998.0, 1994.0, 2002.0],
    3: [1348.0, 1345.0, 1348.0, 1348.0, 1355.0, 1351.0, 1355.0, 1349.0, 1349.0, 1355.0],
    4: [1265.0, 1273.0, 1274.0, 1266.0, 1266.0, 1282.0, 1294.0, 1278.0, 1286.0, 1278.0],
    5: [1310.0, 1300.0, 1340.0, 1335.0, 1310.0, 1310.0, 1325.0, 1300.0, 1320.0, 1282.0],
    6: [1257.0, 1257.0, 1262.0, 1262.0, 1251.0, 1251.0, 1264.0, 1254.0, 1257.0, 1248.0],
}

# Depth data for each k value
depth_data = {
    1: [8730.0, 8797.0, 8823.0, 8663.0, 8761.0, 8719.0, 8726.0, 8754.0, 8753.0, 8757.0],
    2: [8650.0, 8662.0, 8694.0, 8661.0, 8686.0, 8601.0, 8666.0, 8650.0, 8659.0, 8660.0],
    3: [7831.0, 7827.0, 7832.0, 7834.0, 7838.0, 7834.0, 7836.0, 7832.0, 7832.0, 7844.0],
    4: [7783.0, 7795.0, 7792.0, 7786.0, 7764.0, 7785.0, 7803.0, 7790.0, 7787.0, 7788.0],
    5: [7808.0, 7800.0, 7845.0, 7832.0, 7807.0, 7808.0, 7803.0, 7806.0, 7823.0, 7754.0],
    6: [7793.0, 7796.0, 7797.0, 7788.0, 7796.0, 7787.0, 7784.0, 7796.0, 7779.0, 7783.0],
}

# Runtime data
runtime_data = [0.5, 1.2, 3.5, 22, 136, 1010]

# Qiskit baseline (constant across all k values)
qiskit_swaps = 1757.0
qiskit_depth_avg = np.mean([8665.0, 8661.0, 8661.0, 8663.0, 8665.0, 8655.0, 8663.0, 8659.0, 8653.0, 8655.0])

# Calculate statistics
swaps_means = [np.mean(swaps_data[k]) for k in k_values]
swaps_stds = [np.std(swaps_data[k]) for k in k_values]
depth_means = [np.mean(depth_data[k]) for k in k_values]
depth_stds = [np.std(depth_data[k]) for k in k_values]

# Create the figure with subplots (landscape, professional look)
fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

# Subplot 1: Swaps vs k with variance tunnel
ax1 = axs[0]
ax1.fill_between(
    k_values,
    np.array(swaps_means) - np.array(swaps_stds),
    np.array(swaps_means) + np.array(swaps_stds),
    alpha=0.25, color=palette[3], label='±1σ'
)
ax1.plot(k_values, swaps_means, 'o-', linewidth=2.5, markersize=7,
         color=palette[3], label='SABRE (mean)')
ax1.axhline(y=qiskit_swaps, color='gray', linestyle='--', linewidth=2,
            label='Qiskit baseline')
ax1.set_xlabel('Lookahead parameter $k$', fontsize=13, fontweight='bold')
ax1.set_ylabel('Number of SWAP gates', fontsize=13, fontweight='bold')
ax1.set_title('SWAP Gate Count vs $k$', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower left', frameon=False)
ax1.set_xticks(k_values)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.grid(True, alpha=0.2)

for k, mean_swaps in zip(k_values, swaps_means):
    improvement = ((qiskit_swaps - mean_swaps) / qiskit_swaps) * 100
    if improvement > 0:
        ax1.annotate(f'{improvement:.1f}%', (k, mean_swaps),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=10, ha='center', color='green')

# Subplot 2: Circuit Depth vs k with variance tunnel
ax2 = axs[1]
ax2.fill_between(
    k_values,
    np.array(depth_means) - np.array(depth_stds),
    np.array(depth_means) + np.array(depth_stds),
    alpha=0.25, color=palette[6], label='±1σ'
)
ax2.plot(k_values, depth_means, 's-', linewidth=2.5, markersize=7,
         color=palette[6], label='SABRE (mean)')
ax2.axhline(y=qiskit_depth_avg, color='gray', linestyle='--', linewidth=2,
            label='Qiskit baseline')
ax2.set_xlabel('Lookahead parameter $k$', fontsize=13, fontweight='bold')
ax2.set_ylabel('Circuit depth', fontsize=13, fontweight='bold')
ax2.set_title('Circuit Depth vs $k$', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='lower left', frameon=False)
ax2.set_xticks(k_values)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.grid(True, alpha=0.2)

# Subplot 3: Runtime vs k (log scale)
ax3 = axs[2]
ax3.semilogy(k_values, runtime_data, 'D-', linewidth=2.5, markersize=7,
             color=palette[9], label='Runtime')
ax3.set_xlabel('Lookahead parameter $k$', fontsize=13, fontweight='bold')
ax3.set_ylabel('Runtime (s, log scale)', fontsize=13, fontweight='bold')
ax3.set_title('Runtime vs $k$', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11, loc='lower right', frameon=False)
ax3.set_xticks(k_values)
ax3.tick_params(axis='both', which='major', labelsize=11)
ax3.grid(True, alpha=0.2)

for k, runtime in zip(k_values, runtime_data):
    ax3.annotate(f'{runtime}s', (k, runtime),
                 xytext=(0, 10), textcoords='offset points',
                 fontsize=10, ha='center', color=palette[9])

plt.tight_layout()
plt.subplots_adjust(top=0.82)

fig.suptitle(
    'SABRE Algorithm: Lookahead Parameter Analysis\n'
    'Circuit: ham_BH_D_1_d_4_bh_graph_1D_grid (48 qubits, 11,904 DAG nodes)',
    fontsize=16, fontweight='bold'
)

plt.show()
# Optional: Save the figure
# fig.savefig('sabre_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
# fig.savefig('sabre_analysis.pdf', bbox_inches='tight', facecolor='white')