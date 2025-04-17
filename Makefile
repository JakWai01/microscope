run:
	python3 src/microscope/main.py -f examples/sat_n7.qasm

build:
	maturin develop

format:
	black .

lint:
	pyflakes src/

all: build run
