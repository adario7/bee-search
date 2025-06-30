FROM rust:latest

WORKDIR /app
RUN apt-get update && apt-get install -y mold && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release

CMD ["./build/release/bee-search"]
