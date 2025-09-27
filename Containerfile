FROM ubuntu:24.04

WORKDIR /app

# Copy project
ADD . .

# Install deps
RUN apt update && apt install -y \
    make python3-pip python3-venv python3-dev git curl \
    build-essential pkg-config libssl-dev

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create venv
RUN python3 -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install deps including maturin
RUN pip install --upgrade pip && \
    pip install maturin && \
    pip install -r requirements.txt

# Build wheel and install it
RUN maturin build --release -o dist && \
    pip install dist/*.whl

# Run make
RUN make
