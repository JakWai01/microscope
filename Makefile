run:
	python3 src/microscope/main.py -f examples/adder_n10.qasm

build:
	maturin develop

format:
	black .

all: build run

