#!/usr/bin/env python3
"""
plot_denseness_vs_improvement.py

Plot how circuit denseness (depth / qubits) relates to the improvement of
K-SWAP SABRE versus Qiskit.

Positional arguments:
    kswap_file    JSON file with benchmark results for K-SWAP SABRE
    qiskit_file   JSON file with benchmark results for Qiskit

Example:
    python plot_denseness_vs_improvement.py kswap.json qiskit.json \
        --kswap-label "K-SWAP SABRE" --qiskit-label "Qiskit" --out compare.png

Expected JSON structure (top-level dict with "benchmarks" list):
{
  "benchmarks": [
     {
       "name": "...",
       "topology": "square",
       "qubits": 24,
       "swap_stats": {"count": 10, "average": 123, "median": 120, ...},
       "depth": 234
     },
     ...
  ]
}

Notes / choices:
 - Matching of benchmarks uses the tuple (name, topology, qubits).
 - Swap count uses swap_stats['average'] if present, else 'median', else 'count'.
 - Points with depth == 0 or swap == 0 in EITHER dataset are excluded.
 - Improvement reported by default is relative (%) = 100*(qiskit - kswap)/qiskit.
 - Crest colors from seaborn are used for topology coloring.
"""
import json
import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def load_benchmarks(path):
    """Load JSON and return dict keyed by (name, topology, qubits) -> record."""
    with open(path, 'r', encoding='utf8') as f:
        obj = json.load(f)
    if 'benchmarks' not in obj:
        raise ValueError(f"File {path} does not contain top-level 'benchmarks' list.")
    mapping = {}
    for rec in obj['benchmarks']:
        name = rec.get('name')
        topo = rec.get('topology')
        qubits = rec.get('qubits')
        depth = rec.get('depth', None)
        swap_stats = rec.get('swap_stats', {}) or {}
        # prefer 'average', fallback to 'median', then to raw 'count' (if user used that)
        swap_avg = swap_stats.get('average', swap_stats.get('median', swap_stats.get('count', None)))
        key = (name, topo, qubits)
        mapping[key] = {
            'name': name,
            'topology': topo,
            'qubits': qubits,
            'depth': depth,
            'swap_avg': swap_avg,
            'raw': rec
        }
    return mapping

def prepare_dataframe(kswap_map, qiskit_map):
    """Return DataFrame of matched benchmarks (filtered)."""
    keys = set(kswap_map.keys()) & set(qiskit_map.keys())
    rows = []
    for k in keys:
        a = kswap_map[k]
        b = qiskit_map[k]
        # require numeric values
        try:
            depth = float(a['depth'])
            qubits = int(a['qubits'])
            kswap_swaps = float(a['swap_avg'])
            qiskit_swaps = float(b['swap_avg'])
        except Exception:
            continue
        # exclude zeros and invalids as requested
        if depth <= 0 or kswap_swaps <= 0 or qiskit_swaps <= 0 or qubits <= 0:
            continue
        denseness = depth / qubits
        rel_impr = 100.0 * (qiskit_swaps - kswap_swaps) / qiskit_swaps
        abs_impr = qiskit_swaps - kswap_swaps
        rows.append({
            'name': a['name'],
            'topology': a['topology'],
            'qubits': qubits,
            'depth': depth,
            'denseness': denseness,
            'kswap_swaps': kswap_swaps,
            'qiskit_swaps': qiskit_swaps,
            'rel_impr_pct': rel_impr,
            'abs_impr': abs_impr
        })
    df = pd.DataFrame(rows)
    return df

def make_plot(df, kswap_label, qiskit_label, out_path=None, show=True):
    """Create the scatter plot: x=denseness, y=relative improvement (%)"""
    if df.empty:
        raise ValueError("No matched datapoints after filtering (depth==0 or swap==0 removed).")
    sns.set_theme(style="whitegrid")
    topologies = sorted(df['topology'].unique())
    palette = sns.color_palette("crest", n_colors=max(3, len(topologies)))  # crest palette
    topo_color = {t: palette[i % len(palette)] for i, t in enumerate(topologies)}

    # marker sizes proportional to qubit count (scaled)
    qmin, qmax = df['qubits'].min(), df['qubits'].max()
    def sizemap(q):
        # scale to [30, 350]
        if qmax == qmin:
            return 80
        return 30 + ( (q - qmin) / (qmax - qmin) ) * 320

    plt.figure(figsize=(9,6))
    ax = plt.gca()
    for topo in topologies:
        sub = df[df['topology'] == topo]
        sizes = sub['qubits'].apply(sizemap)
        ax.scatter(sub['denseness'], sub['rel_impr_pct'],
                   s=sizes, alpha=0.85, label=topo,
                   color=topo_color[topo], edgecolors='k', linewidths=0.3)

    # horizontal zero line
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Denseness (depth / qubits)', fontsize=12)
    ax.set_ylabel(f"Relative swap reduction (%) of {kswap_label} vs {qiskit_label}", fontsize=12)
    ax.set_title(f"Denseness vs Relative swap reduction ({len(df)} matched circuits)", fontsize=13)

    # topology legend (colors)
    topo_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=topo_color[t],
                           markersize=9, label=t) for t in topologies]
    leg1 = ax.legend(handles=topo_handles, title='Topology', loc='upper right', bbox_to_anchor=(1,1))
    ax.add_artist(leg1)

    # size legend for qubit counts
    # pick representative sizes (min, median, max)
    q_med = int(df['qubits'].median())
    size_vals = [int(qmin), q_med, int(qmax)]
    size_handles = []
    for q in size_vals:
        size_handles.append(plt.scatter([], [], s=sizemap(q),
                                        color='gray', alpha=0.6, edgecolors='k'))
    labels = [f"{q} qubits" for q in size_vals]
    ax.legend(size_handles, labels, title='Qubit count (marker size)', loc='lower right', bbox_to_anchor=(1,0))

    # annotate a bit: mean improvement
    mean_rel = df['rel_impr_pct'].mean()
    ax.text(0.02, 0.02, f"Mean relative improvement: {mean_rel:.1f}%\nPoints: {len(df)}",
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Saved figure to {out_path}")
    if show:
        plt.show()
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Plot denseness vs K-SWAP improvement over Qiskit.")
    p.add_argument("kswap_file", help="JSON results file for K-SWAP SABRE")
    p.add_argument("qiskit_file", help="JSON results file for Qiskit")
    p.add_argument("--kswap-label", default="K-SWAP", help="Label for K-SWAP dataset (used in title/axis)")
    p.add_argument("--qiskit-label", default="Qiskit", help="Label for Qiskit dataset (used in title/axis)")
    p.add_argument("--out", default=None, help="Output file path for the figure (e.g. fig.png). If omitted, the plot is shown but not saved.")
    p.add_argument("--metric", choices=['relative', 'absolute'], default='relative',
                   help="Which improvement metric to plot (default: relative percent).")
    args = p.parse_args()

    if not os.path.exists(args.kswap_file):
        raise FileNotFoundError(args.kswap_file)
    if not os.path.exists(args.qiskit_file):
        raise FileNotFoundError(args.qiskit_file)

    kmap = load_benchmarks(args.kswap_file)
    qmap = load_benchmarks(args.qiskit_file)
    df = prepare_dataframe(kmap, qmap)
    if df.empty:
        raise SystemExit("No matched datapoints left after filtering (depth==0 or swap==0 removed).")

    # if user asked for absolute metric, override plotting values
    if args.metric == 'absolute':
        df['rel_impr_pct'] = df['abs_impr']  # repurpose column for plotting

    make_plot(df, args.kswap_label, args.qiskit_label, out_path=args.out, show=True)

if __name__ == "__main__":
    main()
