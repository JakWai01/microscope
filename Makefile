run: build
	python3 src/microscope/main.py microbench examples/adder_n64.qasm

build:
	maturin develop --release

format:
	black . && pushd rust > /dev/null && cargo fmt --all && popd > /dev/null

lint:
	pyflakes src/

all: build format run

big: build
	python3 src/microscope/main.py microbench ~/Documents/hamiltonians/ham_heis_graph_2D_grid_pbc_qubitnodes_Lx_5_Ly_186_h_3.qasm

big-baseline: build
	python3 src/microscope/main.py baseline ~/Documents/hamiltonians/ham_heis_graph_2D_grid_pbc_qubitnodes_Lx_5_Ly_186_h_3.qasm

small: build
	python3 src/microscope/main.py microbench examples/adder_n10.qasm

profile: build
	perf record -g --call-graph=dwarf -- python3 src/microscope/main.py microbench ~/Documents/hamiltonians/ham_heis_graph_2D_grid_pbc_qubitnodes_Lx_5_Ly_186_h_3.qasm && flamegraph --perfdata perf.data
