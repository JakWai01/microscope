run:
	python3 src/microscope/main.py -f examples/sat_n7.qasm

build:
	maturin develop

format:
	black .

all: build run
