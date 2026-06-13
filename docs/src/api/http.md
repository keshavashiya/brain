# HTTP API

Brain exposes a REST API on port **19789** (default) for all operations.

## Authentication

All endpoints (except `/health`) require an API key via the `Authorization: Bearer <key>` header. The key is generated at `brain init` and stored in `~/.brain/config.yaml`.

## Endpoints

### Health

```http
GET /health
```
Returns `200 OK` when the daemon is running.

### Signals

```http
POST /v1/signals
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "content": "Remember I like Rust",
  "namespace": "personal",
  "source": "curl",
  "channel": "http",
  "sender": "tester"
}
```

### Memory

```http
POST /v1/memory/search
Authorization: Bearer <api_key>

{
  "query": "Rust",
  "namespace": "personal",
  "top_k": 5
}

POST /v1/memory/store
Authorization: Bearer <api_key>

{
  "content": "User prefers dark mode",
  "namespace": "personal",
  "kind": "fact"
}
```

### Webhooks

```http
POST /v1/webhooks/:id
```

### UI

```
GET /ui          # Live dashboard
GET /ui/approvals
GET /ui/memory
GET /ui/audit
```

### Metrics

```http
GET /metrics     # Prometheus-formatted metrics
```

## Adapter ports

| Adapter | Port | Host |
|---------|------|------|
| HTTP | 19789 | 127.0.0.1 |
| WebSocket | 19790 | 127.0.0.1 |
| MCP HTTP | 19791 | 127.0.0.1 |
| gRPC | 19792 | 127.0.0.1 |
| Terminal | 19793 | 127.0.0.1 |
