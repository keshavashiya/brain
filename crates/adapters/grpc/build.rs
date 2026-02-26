fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build server stubs only (skip client to avoid method name conflicts).
    tonic_prost_build::configure()
        .build_client(false)
        .build_server(true)
        .compile_protos(
            &["proto/memory.proto", "proto/agent.proto"],
            &["proto"],
        )?;
    Ok(())
}
