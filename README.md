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

## Requirements

- **Python** 3.10 or later  
- **Rust** (via [rustup](https://rustup.rs/))  
- **maturin** for building Python ↔ Rust extensions  
- **make** (if using a Makefile)  

---

## Quickstart

If you just want to try it out in Docker:

```bash
docker build -t microscope .
````

---

##  Local Development Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/myproject.git
   cd myproject
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install maturin
   pip install -r requirements.txt
   ```

4. **Build and install the Rust extension**

   ```bash
   maturin develop --release
   ```

5. **Run the project**

   ```bash
   make
   ```

---

## Docker Build (Alternative)

If you don’t want to install Rust locally, build inside Docker:

```bash
docker build -t microscope .
```

---

## Notes

* Use `maturin develop` for **editable installs** while developing.
* For a **production build**, create a wheel and install it:

  ```bash
  maturin build --release -o dist
  pip install dist/*.whl
  ```

---

## Contributing

1. Fork the repo
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Make changes
4. Commit (`git commit -m "Add my feature"`)
5. Push (`git push origin feature/my-feature`)
6. Open a Pull Request

---

## License

[Apache 2.0](./LICENSE)
