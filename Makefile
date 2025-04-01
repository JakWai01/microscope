run:
	python3 src/microscope/main.py -f examples/adder_n4.qasm

build:
	maturin develop

format:
	black .

all: format build run

