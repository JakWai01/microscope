run:
	python3 src/microscope/main.py -f examples/adder_n4.qasm

build:
	maturin develop

all: build run 

