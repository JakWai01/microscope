import orjson
import glob
import statistics
from collections import defaultdict

# Input: n accumulated files
input_files = glob.glob("/home/jakob/Documents/Projects/microscope/assets/benchmark/test-k4/combined/*")

# Collect data
bench_swap_data = defaultdict(list)
bench_topo_data = defaultdict(str)

for path in input_files:
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    for bench in data.get("benchmarks", []):
        swaps = bench.get("extra_info", {}).get("output_circuit_operations", {}).get("swap")
        topology = bench.get("params", {}).get("circ_and_topo", [])[1]

        if swaps is not None:
            bench_swap_data[bench["name"]].append(swaps)
            bench_topo_data[bench["name"]] = topology

# Build aggregated results
aggregated = {"benchmarks": []}
for name, swaps in bench_swap_data.items():
    avg = statistics.mean(swaps)
    med = statistics.median(swaps)
    mn = min(swaps)
    mx = max(swaps)
    aggregated["benchmarks"].append({
        "name": name,
        "topology": bench_topo_data[name],
        "swap_stats": {
            "count": len(swaps),
            "average": avg,
            "median": med,
            "min": mn,
            "max": mx,
        }
    })

# Write to file
output_path = "/home/jakob/Documents/Projects/microscope/assets/benchmark/test-k4/stats/benchmark_swap_stats.json"
with open(output_path, "wb") as f:
    f.write(orjson.dumps(aggregated, option=orjson.OPT_INDENT_2))

print(f"Wrote aggregated swap stats for {len(aggregated['benchmarks'])} benchmarks to {output_path}")
