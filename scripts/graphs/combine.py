import orjson
import glob

# List of input JSON files (adjust the pattern or provide manually)
input_files = glob.glob("/home/jakob/Documents/Projects/microscope/assets/benchmark/*test_k1.json.json")

# The combined structure will mimic pytest-benchmark schema
combined = {
    "machine_info": None,
    "commit_info": None,
    "benchmarks": []
}

for i, path in enumerate(sorted(input_files)):
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    if i == 0:
        # Take machine_info + commit_info from the first file
        combined["machine_info"] = data.get("machine_info")
        combined["commit_info"] = data.get("commit_info")
    # Always extend the benchmarks
    combined["benchmarks"].extend(data.get("benchmarks", []))

# Write the merged JSON file
output_path = "/home/jakob/Documents/Projects/microscope/assets/benchmark/test_k1_combined.json"
with open(output_path, "wb") as f:
    f.write(orjson.dumps(combined, option=orjson.OPT_INDENT_2))

print(f"Combined {len(input_files)} files into {output_path}")
