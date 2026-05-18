fn main() {
    tonic_build::configure()
        .compile_protos(&["proto/federated_ai.proto"], &["proto"])
        .unwrap();
}
