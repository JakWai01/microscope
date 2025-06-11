bench: build
	python3 src/microscope/main.py microbench

build:
	maturin develop --release

format:
	black . && pushd rust > /dev/null && cargo fmt --all && popd > /dev/null

base: build
	python3 src/microscope/main.py baseline

profile: build
	perf record -g --call-graph=dwarf -- python3 src/microscope/main.py microbench && flamegraph --perfdata perf.data

docs:
	pushd rust && cargo doc --document-private-items --open && popd