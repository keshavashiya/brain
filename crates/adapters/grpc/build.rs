fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Vendor protoc so builds work without a system-installed protobuf compiler.
    std::env::set_var("PROTOC", protobuf_src::protoc());

    // Build server stubs only (skip client to avoid method name conflicts).
    tonic_prost_build::configure()
        .build_client(false)
        .build_server(true)
        .compile_protos(&["proto/memory.proto", "proto/agent.proto"], &["proto"])?;
    Ok(())
}
