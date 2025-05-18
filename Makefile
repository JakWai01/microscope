run:
	python3 src/microscope/main.py --plot True examples/adder_n28.qasm examples/adder_n10.qasm examples/adder_n4.qasm
build:
	maturin develop

format:
	black .

lint:
	pyflakes src/

all: build run
