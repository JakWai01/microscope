run:
	python3 src/microscope/main.py ham_heis_graph_2D_grid_pbc_qubitnodes_Lx_5_Ly_186_h_3.qasm
build:
	maturin develop

format:
	black .

lint:
	pyflakes src/

all: build run
