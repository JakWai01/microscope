run: build
	python3 src/microscope/main.py microbench examples/adder_n4.qasm examples/adder_n10.qasm examples/adder_n28.qasm

build:
	maturin build -r

format:
	black . && pushd rust > /dev/null && cargo fmt --all -- --check && popd > /dev/null

lint:
	pyflakes src/

all: build run