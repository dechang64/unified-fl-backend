# ── Stage 1: Build ──
FROM rust:1.95-slim AS builder

RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock build.rs ./
COPY proto ./proto
COPY src ./src

RUN cargo build --release

# ── Stage 2: Runtime ──
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/unified-fl-backend /usr/local/bin/
COPY --from=builder /app/proto ./proto

RUN mkdir -p /data

ENV RUST_LOG=info
EXPOSE 50051 8080

ENTRYPOINT ["unified-fl-backend"]
CMD ["--dimension", "384", "--grpc-port", "50051", "--http-port", "8080", "--db-path", "/data/vectors.db"]
