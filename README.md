# microscope

This repository contains the implementation of *K-SWAP SABRE*, an extension to
the classical SABRE algorithm developed as part of my master thesis.

In order to transpile quantum circuits, so-called SWAP gates need to be inserted
into the circuit in order to support two-qubit operations between two logical
qubits. During the process of inserting SWAPs, the precedence relations defined
in the DAG have to be respected.

## The Qubit Mapping Problem

The following circuit resembles a 4-qubit adder that is given as an input
circuit.

![Input Circuit](./assets/img/input_circuit.png)

The circuit below shows the result when transpiling the circuit using *K-SWAP SABRE*
algorithm based on a trivial initial mapping.

![Output Circuit Transpiled using K-SWAP SABRE](./assets/img/sabre_circuit.png)

## Setup

Make sure to install the [Rust](https://www.rust-lang.org/tools/install) programming language. Then, install all the required Python dependencies.

```
pip install -r requirements.txt
```

## Run 

To run the circuits configured in `config.yaml`, execute:

```
make
```