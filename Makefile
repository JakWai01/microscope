ocular: build
	python3 src/microscope/main.py ocular

olsq: build
	python3 src/microscope/main.py olsq

build:
	MALLOC_CONF="thp:always,metadata_thp:always" maturin develop --release

lint:
	pushd rust > /dev/null && cargo clippy && popd

format:
	black . && pushd rust > /dev/null && cargo fmt --all && popd > /dev/null

profile: build
	perf record -g --call-graph=dwarf -- python3 src/microscope/main.py ocular && flamegraph --perfdata perf.data

docs:
	pushd rust && cargo doc --document-private-items --open && popd
