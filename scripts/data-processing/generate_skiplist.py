import json

def filter_tests_by_qubits(file_path, x, y, output_file):
    """
    Write full-names of tests where input_num_qubits is > x and < y to a file.

    Args:
        file_path (str): Path to the JSON file.
        x (int): Lower bound (exclusive).
        y (int): Upper bound (exclusive).
        output_file (str): Path to the output text file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])
    results = []

    for bench in benchmarks:
        extra_info = bench.get("extra_info", {})
        num_qubits = extra_info.get("input_num_qubits")
        if num_qubits is not None and x <= num_qubits <= y:
            results.append(bench.get("fullname"))

    with open(output_file, "w") as f:
        for r in results:
            f.write(r + "\n")

    print(f"Wrote {len(results)} matching full-names to {output_file}")


if __name__ == "__main__":
    # Example usage
    file_path = "../assets/benchmark/test-k1-1h/combined/test_k1_1h_trial_1_combined.json"
    x = 0   # lower bound
    y = 40  # upper bound
    output_file = "skiplist.txt"

    filter_tests_by_qubits(file_path, x, y, output_file)
