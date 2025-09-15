#!/usr/bin/env python3
"""
plot_improvement_violin_square.py

Produce a violin plot of relative improvement (%) of "ocular" (K-SWAP SABRE k=3)
compared to Qiskit for circuits with qubits <= 40, restricted to square topology.
"""

import json
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_benchmarks(path):
    with open(path, 'r') as f:
        j = json.load(f)
    return j.get('benchmarks', [])

def make_map(benchmarks):
    d = {}
    for b in benchmarks:
        name = b.get('name')
        if name is None:
            continue
        swap_stats = b.get('swap_stats', {}) or {}
        avg = swap_stats.get('min')
        qubits = b.get('qubits')
        topology = b.get('topology')
        d[name] = {'avg': avg, 'qubits': qubits, 'topology': topology}
    return d

def plot_violin_split(df, out_path, title=None):
    if df.empty:
        raise ValueError("No matched benchmarks to plot (empty dataframe).")

    sns.set_theme(style='white', context='paper', rc={
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Liberation Serif'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
    })
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    unique_benchmarks = df['benchmark'].nunique()

    if unique_benchmarks == 1:
        # Just one benchmark → full violin
        ax = sns.violinplot(
            y='improvement',
            data=df,
            inner='box',
            # cut=0,
            scale='area',
            bw=0.3,
            width=0.7,
            color='#4477AA',
            linewidth=1.2,
        )
    else:
        # Multiple benchmarks → split violin
        ax = sns.violinplot(
            y='improvement',
            hue='benchmark',
            data=df,
            split=True,
            inner='box',
            # cut=0,
            scale='area',
            bw=0.3,
            width=0.7,
            palette=['#4477AA', '#EE7733'],
            linewidth=1.2,
        )

    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1, zorder=0)

    min_y = df['improvement'].min()
    max_y = df['improvement'].max()
    pad = (max_y - min_y) * 0.1 if max_y > min_y else 5
    ax.set_ylim(min_y - pad, max_y + pad)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, prune='both', integer=False))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Print stats
    for label in df['benchmark'].unique():
        sub = df[df['benchmark'] == label]
        n = len(sub)
        median = sub['improvement'].median()
        mean = sub['improvement'].mean()
        min_val = sub['improvement'].min()
        max_val = sub['improvement'].max()
        print(
            f"Stats for {label}:\n"
            f"  n      = {n}\n"
            f"  median = {median:.1f} %\n"
            f"  mean   = {mean:.1f} %\n"
            f"  min    = {min_val:.1f} %\n"
            f"  max    = {max_val:.1f} %\n"
        )

    ax.set_ylabel('Relative Improvement (%)')
    ax.set_xlabel('')
    # ax.set_title(title or 'Relative Improvement of K-SWAP SABRE vs Qiskit SABRE (IQR) \n(square topology, k=3, ≤ 1000 qubits)')
    ax.set_title(
        (title or 'Relative Improvement of K-SWAP SABRE vs Qiskit SABRE \n'
                '(square topology, k=3, ≤ 1000 qubits, IQR)')
    )
    # ax.annotate(
    #     "Outliers Removed via IQR rule",
    #     xy=(0.5, -0.15),
    #     xycoords='axes fraction',
    #     ha='center',
    #     fontsize=9,
    #     style='italic'
    # )
    ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='gray', alpha=0.5)
    ax.xaxis.grid(False)

    # Only add legend if multiple benchmarks
    if unique_benchmarks > 1:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Benchmark', loc='upper right', frameon=True)
    else:
        ax.get_legend().remove() if ax.get_legend() else None

    plt.tight_layout()
    ax.get_figure().savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(ax.get_figure())

def compute_improvements(ocular_map, qiskit_map, min_qubits=0, max_qubits=40, topology_filter="square"):
    rows = []
    for name, o in ocular_map.items():
        if name not in qiskit_map:
            continue
        q = qiskit_map[name]

        # enforce topology filter
        topo = o.get('topology') or q.get('topology')
        if topology_filter and topo != topology_filter:
            continue

        o_avg = o.get('avg')
        q_avg = q.get('avg')
        qubits = o.get('qubits') if o.get('qubits') is not None else q.get('qubits')

        if qubits is None:
            continue
        try:
            qubits_int = int(qubits)
        except Exception:
            continue
        if qubits_int < min_qubits or qubits_int > max_qubits:
            continue

        if q_avg is None or q_avg == 0 or o_avg is None:
            continue

        improvement = (q_avg - o_avg) / q_avg * 100.0
        rows.append({
            'name': name,
            'qubits': qubits_int,
            'topology': topo,
            'ocular_avg': float(o_avg),
            'qiskit_avg': float(q_avg),
            'improvement': float(improvement)
        })
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser(description="Violin plot of relative improvement (ocular vs qiskit) for square topology")
    p.add_argument('--ocular', type=Path, default=Path('/mnt/data/ocular_benchmark_swap_stats.json'))
    p.add_argument('--ocular2', type=Path, help="Second ocular benchmark file (different k)", default=None)
    p.add_argument('--qiskit', type=Path, default=Path('/mnt/data/qiskit_benchmark_swap_stats.json'))
    p.add_argument('--min-qubits', type=int, default=0, help="Minimum number of qubits (inclusive)")
    p.add_argument('--max-qubits', type=int, default=40, help="Maximum number of qubits (inclusive)")
    p.add_argument('--out', type=Path, default=Path('violin_improvement_square_0_40.png'))
    args = p.parse_args()

    qiskit_bench = load_benchmarks(args.qiskit)
    qiskit_map = make_map(qiskit_bench)

    # First ocular
    ocular_bench = load_benchmarks(args.ocular)
    ocular_map = make_map(ocular_bench)
    df1 = compute_improvements(
        ocular_map, qiskit_map,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        topology_filter="square"
    )
    df1['benchmark'] = 'k = 3'  # or use a label that matches your config

    dfs = [df1]

    # Second ocular (optional)
    if args.ocular2:
        ocular2_bench = load_benchmarks(args.ocular2)
        ocular2_map = make_map(ocular2_bench)
        df2 = compute_improvements(
            ocular2_map, qiskit_map,
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
            topology_filter="square"
        )
        df2['benchmark'] = 'k = 4'  # or use a label that matches your config
        dfs.append(df2)

    df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        print("No matched benchmarks found for square topology within qubit range.")
        return

    csv_out = args.out.with_suffix('.csv')
    df.to_csv(csv_out, index=False)
    print(f"Saved computed improvements to {csv_out}")

    # lower = df['improvement'].quantile(0.01)
    # upper = df['improvement'].quantile(0.99)

    # df = df[(df['improvement'] >= lower) & (df['improvement'] <= upper)]
    Q1 = df['improvement'].quantile(0.25)
    Q3 = df['improvement'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df['improvement'] >= lower) & (df['improvement'] <= upper)]

    plot_violin_split(df, args.out)
    print(f"Violin plot saved to {args.out}")

if __name__ == '__main__':
    main()
