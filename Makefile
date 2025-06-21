ocular: build
	python3 src/microscope/main.py ocular

bench: build
	python3 src/microscope/main.py bench 

single: build
	RUST_BACKTRACE=1 python3 src/microscope/main.py single

build:
	maturin develop --release

format:
	black . && pushd rust > /dev/null && cargo fmt --all && popd > /dev/null

base: build
	python3 src/microscope/main.py baseline

profile: build
	perf record -g --call-graph=dwarf -- python3 src/microscope/main.py bench && flamegraph --perfdata perf.data

docs:
	pushd rust && cargo doc --document-private-items --open && popd
