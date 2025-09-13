import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Set the style for scientific plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data from your SABRE implementation
k_values = [1, 2, 3, 4, 5]

# Swap counts for each k value (10 runs each)
swaps_data = {
    1: [2018.0, 2085.0, 2092.0, 1973.0, 2053.0, 2017.0, 2025.0, 2037.0, 2037.0, 2055.0],
    2: [1990.0, 2008.0, 2022.0, 2004.0, 2016.0, 1948.0, 2000.0, 1998.0, 1994.0, 2002.0],
    3: [1348.0, 1345.0, 1348.0, 1348.0, 1355.0, 1351.0, 1355.0, 1349.0, 1349.0, 1355.0],
    4: [1265.0, 1273.0, 1274.0, 1266.0, 1266.0, 1282.0, 1294.0, 1278.0, 1286.0, 1278.0],
    5: [1310.0, 1300.0, 1340.0, 1335.0, 1310.0, 1310.0, 1325.0, 1300.0, 1320.0, 1282.0]
}

# Depth data for each k value
depth_data = {
    1: [8730.0, 8797.0, 8823.0, 8663.0, 8761.0, 8719.0, 8726.0, 8754.0, 8753.0, 8757.0],
    2: [8650.0, 8662.0, 8694.0, 8661.0, 8686.0, 8601.0, 8666.0, 8650.0, 8659.0, 8660.0],
    3: [7831.0, 7827.0, 7832.0, 7834.0, 7838.0, 7834.0, 7836.0, 7832.0, 7832.0, 7844.0],
    4: [7783.0, 7795.0, 7792.0, 7786.0, 7764.0, 7785.0, 7803.0, 7790.0, 7787.0, 7788.0],
    5: [7808.0, 7800.0, 7845.0, 7832.0, 7807.0, 7808.0, 7803.0, 7806.0, 7823.0, 7754.0]
}

# Runtime data
runtime_data = [0.5, 1.2, 3.5, 22, 136]

# Qiskit baseline (constant across all k values)
qiskit_swaps = 1757.0
qiskit_depth_avg = np.mean([8665.0, 8661.0, 8661.0, 8663.0, 8665.0, 8655.0, 8663.0, 8659.0, 8653.0, 8655.0])

# Calculate statistics
swaps_means = [np.mean(swaps_data[k]) for k in k_values]
swaps_stds = [np.std(swaps_data[k]) for k in k_values]
depth_means = [np.mean(depth_data[k]) for k in k_values]
depth_stds = [np.std(depth_data[k]) for k in k_values]

# Create the figure with subplots
fig = plt.figure(figsize=(16, 10))

# Create a 2x2 grid with the bottom subplot spanning both columns
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)

# Subplot 1: Swaps vs k with variance tunnel
ax1 = fig.add_subplot(gs[0, 0])

# Plot variance tunnel (shaded region)
ax1.fill_between(k_values, 
                 np.array(swaps_means) - np.array(swaps_stds), 
                 np.array(swaps_means) + np.array(swaps_stds), 
                 alpha=0.3, color='steelblue', label='±1σ variance')

# Plot mean line
ax1.plot(k_values, swaps_means, 'o-', linewidth=2.5, markersize=8, 
         color='steelblue', label='SABRE (mean)')

# Add Qiskit baseline
ax1.axhline(y=qiskit_swaps, color='red', linestyle='--', linewidth=2, 
            label='Qiskit baseline')

ax1.set_xlabel('Lookahead Parameter (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of SWAP Gates', fontsize=12, fontweight='bold')
ax1.set_title('SWAP Gate Optimization vs Lookahead Depth', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xticks(k_values)

# Add improvement percentages
for i, (k, mean_swaps) in enumerate(zip(k_values, swaps_means)):
    improvement = ((qiskit_swaps - mean_swaps) / qiskit_swaps) * 100
    if improvement > 0:
        ax1.annotate(f'{improvement:.1f}% better', 
                    (k, mean_swaps), xytext=(5, 10), 
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Subplot 2: Circuit Depth vs k with variance tunnel
ax2 = fig.add_subplot(gs[0, 1])

# Plot variance tunnel
ax2.fill_between(k_values, 
                 np.array(depth_means) - np.array(depth_stds), 
                 np.array(depth_means) + np.array(depth_stds), 
                 alpha=0.3, color='darkorange', label='±1σ variance')

# Plot mean line
ax2.plot(k_values, depth_means, 's-', linewidth=2.5, markersize=8, 
         color='darkorange', label='SABRE (mean)')

# Add Qiskit baseline
ax2.axhline(y=qiskit_depth_avg, color='red', linestyle='--', linewidth=2, 
            label='Qiskit baseline')

ax2.set_xlabel('Lookahead Parameter (k)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Circuit Depth', fontsize=12, fontweight='bold')
ax2.set_title('Circuit Depth Optimization vs Lookahead Depth', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xticks(k_values)

# Subplot 3: Runtime vs k (log scale)
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogy(k_values, runtime_data, 'D-', linewidth=2.5, markersize=8, 
             color='purple', label='Runtime')
ax3.set_xlabel('Lookahead Parameter (k)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Runtime (seconds, log scale)', fontsize=12, fontweight='bold')
ax3.set_title('Computational Cost vs Lookahead Depth', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(k_values)

# Add runtime annotations
for i, (k, runtime) in enumerate(zip(k_values, runtime_data)):
    ax3.annotate(f'{runtime}s', 
                (k, runtime), xytext=(0, 15), 
                textcoords='offset points', ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.7))

# Subplot 4: Efficiency Analysis (Quality vs Speed Trade-off)
ax4 = fig.add_subplot(gs[1, 1])

# Calculate efficiency metrics
swap_improvement = [(qiskit_swaps - mean) / qiskit_swaps * 100 for mean in swaps_means]
depth_improvement = [(qiskit_depth_avg - mean) / qiskit_depth_avg * 100 for mean in depth_means]

# Scatter plot with size proportional to runtime
sizes = [20 + 5 * np.log10(rt) for rt in runtime_data]
scatter = ax4.scatter(swap_improvement, depth_improvement, s=sizes, 
                     c=k_values, cmap='viridis', alpha=0.7, edgecolors='black')

# Add labels for each point
for i, k in enumerate(k_values):
    ax4.annotate(f'k={k}', (swap_improvement[i], depth_improvement[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax4.set_xlabel('SWAP Reduction (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Depth Reduction (%)', fontsize=12, fontweight='bold')
ax4.set_title('Quality-Speed Trade-off Analysis', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add colorbar for k values
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Lookahead k', fontsize=11, fontweight='bold')

# Subplot 5: Summary Statistics Table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

# Create summary table
table_data = []
headers = ['k', 'Avg Swaps', 'Swap Improv. (%)', 'Avg Depth', 'Depth Improv. (%)', 'Runtime (s)', 'Efficiency*']

for i, k in enumerate(k_values):
    efficiency = (swap_improvement[i] + depth_improvement[i]) / (np.log10(runtime_data[i] + 1) + 1)
    table_data.append([
        f'{k}',
        f'{swaps_means[i]:.1f}±{swaps_stds[i]:.1f}',
        f'{swap_improvement[i]:.1f}%',
        f'{depth_means[i]:.1f}±{depth_stds[i]:.1f}',
        f'{depth_improvement[i]:.1f}%',
        f'{runtime_data[i]}s',
        f'{efficiency:.2f}'
    ])

# Create table
table = ax5.table(cellText=table_data, colLabels=headers,
                 cellLoc='center', loc='center',
                 colWidths=[0.08, 0.15, 0.12, 0.15, 0.12, 0.1, 0.1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style the table
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code the efficiency column (last column index is len(headers)-1)
efficiency_col_idx = len(headers) - 1
for i in range(1, len(k_values) + 1):
    efficiency_val = float(table_data[i-1][-1])
    if efficiency_val > 15:
        table[(i, efficiency_col_idx)].set_facecolor('#E8F5E8')
    elif efficiency_val > 10:
        table[(i, efficiency_col_idx)].set_facecolor('#FFF3CD')
    else:
        table[(i, efficiency_col_idx)].set_facecolor('#F8D7DA')

ax5.set_title('Performance Summary\n*Efficiency = (Quality Improvement) / (log₁₀(Runtime) + 1)', 
              fontsize=12, fontweight='bold', pad=20)

# Overall figure title
fig.suptitle('SABRE Algorithm: Lookahead Parameter Analysis\n' + 
             'Circuit: ham_BH_D_1_d_4_bh_graph_1D_grid (48 qubits, 11,904 DAG nodes)', 
             fontsize=16, fontweight='bold', y=0.98)

# Add a text box with key insights
insight_text = """Key Insights:
• k=3-4 provides optimal balance of quality and speed
• Significant improvement over Qiskit baseline
• Diminishing returns beyond k=4
• Runtime grows exponentially with k"""

fig.text(0.02, 0.02, insight_text, fontsize=10, 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
         verticalalignment='bottom')

plt.tight_layout()
plt.show()

# Optional: Save the figure
# plt.savefig('sabre_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig('sabre_analysis.pdf', bbox_inches='tight', facecolor='white')