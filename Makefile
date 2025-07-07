ocular: build
	RUST_BACKTRACE=1 python3 src/microscope/main.py ocular

qiskit: build
	RUST_BACKTRACE=1 python3 src/microscope/main.py qiskit

olsq: build
	RUST_BACKTRACE=1 python3 src/microscope/main.py olsq

build:
	maturin develop --release

format:
	black . && pushd rust > /dev/null && cargo fmt --all && popd > /dev/null

profile: build
	perf record -g --call-graph=dwarf -- python3 src/microscope/main.py ocular && flamegraph --perfdata perf.data

docs:
	pushd rust && cargo doc --document-private-items --open && popd
