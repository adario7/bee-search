FROM rust:latest

WORKDIR /app
RUN apt-get update && apt-get install -y git mold && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release

CMD ["sh", "-c", "if [ -f ./build/release/bee-search ]; then ./build/release/bee-search; else ./target/release/bee-search; fi"]
