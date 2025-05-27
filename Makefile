run: build
	python3 src/microscope/main.py --plot True examples/adder_n10.qasm examples/adder_n4.qasm

build:
	maturin develop

format:
	black . && pushd rust > /dev/null && cargo fmt --all -- --check && popd > /dev/null

lint:
	pyflakes src/

all: build run
