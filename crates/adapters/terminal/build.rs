fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("PROTOC", protobuf_src::protoc());

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["proto/terminal.proto"], &["proto"])?;
    Ok(())
}
