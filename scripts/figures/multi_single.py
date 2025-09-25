import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Style ---
sns.set_theme(style='whitegrid')
palette = sns.color_palette("crest", n_colors=3)  # one color per circuit
markers = ['o', 's', '^']  # circle, square, triangle

# --- Data ---
k_values = [1, 2, 3, 4, 5, 6]

# Circuit 1: ham_BH_D_1_d_4
ham_swaps = {
    1: [2018, 2085, 2092, 1973, 2053, 2017, 2025, 2037, 2037, 2055],
    2: [1990, 2008, 2022, 2004, 2016, 1948, 2000, 1998, 1994, 2002],
    3: [1348, 1345, 1348, 1348, 1355, 1351, 1355, 1349, 1349, 1355],
    4: [1265, 1273, 1274, 1266, 1266, 1282, 1294, 1278, 1286, 1278],
    5: [1310, 1300, 1340, 1335, 1310, 1310, 1325, 1300, 1320, 1282],
    6: [1257, 1257, 1262, 1262, 1251, 1251, 1264, 1254, 1257, 1248],
}

ham_depth = {
    1: [8730, 8797, 8823, 8663, 8761, 8719, 8726, 8754, 8753, 8757],
    2: [8650, 8662, 8694, 8661, 8686, 8601, 8666, 8650, 8659, 8660],
    3: [7831, 7827, 7832, 7834, 7838, 7834, 7836, 7832, 7832, 7844],
    4: [7783, 7795, 7792, 7786, 7764, 7785, 7803, 7790, 7787, 7788],
    5: [7808, 7800, 7845, 7832, 7807, 7808, 7803, 7806, 7823, 7754],
    6: [7793, 7796, 7797, 7788, 7796, 7787, 7784, 7796, 7779, 7783],
}

ham_runtime = [0.5, 1.2, 3.5, 22, 136, 1010]

ham_qiskit_swaps_vals = [1757]
ham_qiskit_depth_vals = [8665, 8661, 8661, 8663, 8665, 8655, 8663, 8659, 8653, 8655]
ham_qiskit_swaps = np.mean(ham_qiskit_swaps_vals)
ham_qiskit_swaps_std = np.std(ham_qiskit_swaps_vals)
ham_qiskit_depth = np.mean(ham_qiskit_depth_vals)
ham_qiskit_depth_std = np.std(ham_qiskit_depth_vals)

# Circuit 2: ham_TSP_Ncity_8_tsp
uly_swaps = {
    1: [1011, 1008, 1111, 882, 962, 1062, 1000, 958, 997, 1054],
    2: [711, 814, 864, 874, 757, 887, 760, 749, 832, 884],
    3: [760, 807, 847, 800, 752, 827, 809, 771, 813, 719],
    4: [719, 707, 699, 711, 699, 707, 711, 703, 703, 703],
    5: [640, 635, 629, 659, 609, 649, 629, 610, 705, 594],
    6: [670, 629, 664, 658, 658, 663, 633, 629, 658, 653],
}

uly_depth = {
    1: [2723, 2681, 2819, 2603, 2640, 2783, 2650, 2671, 2647, 2726],
    2: [2470, 2535, 2557, 2614, 2515, 2620, 2446, 2497, 2544, 2637],
    3: [2478, 2496, 2534, 2545, 2473, 2563, 2510, 2471, 2514, 2422],
    4: [2448, 2434, 2422, 2443, 2428, 2433, 2442, 2428, 2429, 2430],
    5: [2393, 2385, 2392, 2419, 2370, 2406, 2391, 2374, 2431, 2348],
    6: [2426, 2377, 2424, 2409, 2409, 2407, 2383, 2377, 2411, 2404],
}

uly_runtime = [0.15, 0.2, 1.27, 15.9, 68.7, 8750]

uly_qiskit_swaps_vals = [607, 584, 574, 575, 552, 518, 550, 557, 526, 529]
uly_qiskit_depth_vals = [2388, 2371, 2386, 2374, 2389, 2374, 2368, 2377, 2358, 2380]
uly_qiskit_swaps = np.mean(uly_qiskit_swaps_vals)
uly_qiskit_swaps_std = np.std(uly_qiskit_swaps_vals)
uly_qiskit_depth = np.mean(uly_qiskit_depth_vals)
uly_qiskit_depth_std = np.std(uly_qiskit_depth_vals)

# Circuit 3: ham_OH_JW10
oh_swaps = {
    1: [763, 677, 741, 743, 705, 679, 673, 657, 753, 715],
    2: [946, 1202, 1106, 1228, 1294, 1348, 1092, 1312, 1340, 1162],
    3: [819, 980, 687, 1048, 723, 885, 993, 1110, 867, 1074],
    4: [827, 915, 939, 930, 923, 843, 962, 918, 883, 967],
    5: [974, 789, 1076, 911, 691, 1061, 799, 699, 1149, 791],
    6: [820, 850, 988, 719, 832, 868, 928, 689, 832, 814],
}

oh_depth = {
    1: [3495, 3434, 3494, 3494, 3455, 3439, 3434, 3427, 3497, 3468],
    2: [3663, 3907, 3814, 3909, 3975, 4034, 3783, 4018, 4006, 3880],
    3: [3547, 3678, 3449, 3757, 3453, 3615, 3702, 3838, 3632, 3812],
    4: [3592, 3647, 3667, 3702, 3688, 3625, 3728, 3663, 3636, 3683],
    5: [3709, 3532, 3787, 3653, 3441, 3788, 3551, 3459, 3875, 3514],
    6: [3588, 3601, 3700, 3475, 3582, 3608, 3655, 3462, 3570, 3568],
}

oh_runtime = [0.21, 0.31, 0.78, 5.79, 33.7, 248]

oh_qiskit_swaps_vals = [813, 705, 745, 735, 743, 751, 695, 868, 814, 743]
oh_qiskit_depth_vals = [3442, 3383, 3393, 3388, 3400, 3419, 3362, 3510, 3489, 3373]
oh_qiskit_swaps = np.mean(oh_qiskit_swaps_vals)
oh_qiskit_swaps_std = np.std(oh_qiskit_swaps_vals)
oh_qiskit_depth = np.mean(oh_qiskit_depth_vals)
oh_qiskit_depth_std = np.std(oh_qiskit_depth_vals)

# Organize into a dict
circuits = {
    "ham_BH_D_1_d_4": (ham_swaps, ham_depth, ham_runtime, ham_qiskit_swaps, ham_qiskit_swaps_std, ham_qiskit_depth, ham_qiskit_depth_std),
    "ham_TSP_Ncity_8_tsp": (uly_swaps, uly_depth, uly_runtime, uly_qiskit_swaps, uly_qiskit_swaps_std, uly_qiskit_depth, uly_qiskit_depth_std),
    "ham_OH_JW10": (oh_swaps, oh_depth, oh_runtime, oh_qiskit_swaps, oh_qiskit_swaps_std, oh_qiskit_depth, oh_qiskit_depth_std),
}

# --- Plot ---
fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

for i, (circuit, (swaps, depths, runtimes, q_swaps, q_swaps_std, q_depth, q_depth_std)) in enumerate(circuits.items()):
    color = palette[i]
    marker = markers[i]

    # Compute means & stds
    swaps_means = [np.mean(swaps[k]) for k in k_values]
    swaps_stds = [np.std(swaps[k]) for k in k_values]
    depth_means = [np.mean(depths[k]) for k in k_values]
    depth_stds = [np.std(depths[k]) for k in k_values]

    # Subplot 1: Swaps
    axs[0].fill_between(k_values,
                        np.array(swaps_means) - np.array(swaps_stds),
                        np.array(swaps_means) + np.array(swaps_stds),
                        alpha=0.15, color=color)
    axs[0].plot(k_values, swaps_means, marker+'-', linewidth=2, markersize=6,
                label=f'{circuit}', color=color)
    axs[0].axhline(y=q_swaps, color=color, linestyle='--', linewidth=1.5)
    axs[0].fill_between(k_values, q_swaps - q_swaps_std, q_swaps + q_swaps_std,
                        color=color, alpha=0.1)

    # Subplot 2: Depth
    axs[1].fill_between(k_values,
                        np.array(depth_means) - np.array(depth_stds),
                        np.array(depth_means) + np.array(depth_stds),
                        alpha=0.15, color=color)
    axs[1].plot(k_values, depth_means, marker+'-', linewidth=2, markersize=6,
                label=f'{circuit}', color=color)
    axs[1].axhline(y=q_depth, color=color, linestyle='--', linewidth=1.5)
    axs[1].fill_between(k_values, q_depth - q_depth_std, q_depth + q_depth_std,
                        color=color, alpha=0.1)

    # Subplot 3: Runtime
    axs[2].semilogy(k_values, runtimes, marker+'-', linewidth=2, markersize=6,
                    label=f'{circuit}', color=color)

# --- Formatting ---
axs[0].set_xlabel('Lookahead parameter $k$', fontsize=13, fontweight='bold')
axs[0].set_ylabel('Number of SWAP gates', fontsize=13, fontweight='bold')
axs[0].set_title('SWAP Gate Count vs $k$', fontsize=14, fontweight='bold')
axs[0].legend(fontsize=10, loc='upper right', frameon=False)

axs[1].set_xlabel('Lookahead parameter $k$', fontsize=13, fontweight='bold')
axs[1].set_ylabel('Circuit depth', fontsize=13, fontweight='bold')
axs[1].set_title('Circuit Depth vs $k$', fontsize=14, fontweight='bold')
axs[1].legend(fontsize=10, loc='upper right', frameon=False)

axs[2].set_xlabel('Lookahead parameter $k$', fontsize=13, fontweight='bold')
axs[2].set_ylabel('Runtime (s, log scale)', fontsize=13, fontweight='bold')
axs[2].set_title('Runtime vs $k$', fontsize=14, fontweight='bold')
axs[2].legend(fontsize=10, loc='upper left', frameon=False)

plt.suptitle('SABRE Algorithm: Lookahead Parameter Analysis Across Circuits',
             fontsize=16, fontweight='bold')
plt.show()