# gRPC API

Brain exposes a gRPC API on port **19792** for the memory service.

The gRPC adapter provides the same memory operations available via HTTP, with the efficiency of binary protobuf serialization. This is the recommended transport for programmatic clients doing high-volume memory operations.
