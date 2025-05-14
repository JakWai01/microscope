run:
	python3 src/microscope/main.py -f examples/adder_n64.qasm -q True

build:
	maturin develop

format:
	black .

lint:
	pyflakes src/

all: build run
