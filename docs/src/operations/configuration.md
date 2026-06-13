# Configuration

Brain's configuration lives in `~/.brain/config.yaml`. Config precedence (highest wins):

1. Env vars prefixed `BRAIN_` with `__` separator (e.g. `BRAIN_LLM__API_KEY=…`)
2. `~/.brain/config.yaml`
3. Embedded defaults (`crates/core/default.yaml`)

## LLM Configuration

**Single provider:**
```yaml
llm:
  provider: "ollama"
  base_url: "http://localhost:11434"
  model: "qwen2.5-coder:7b"
```

**Multi-provider pool (actual default):**
```yaml
llm:
  temperature: 0.7
  max_tokens: 4096
  context_window: 8192
  providers:
    - name: ollama
      kind: ollama
      base_url: "http://localhost:11434"
      model: "qwen2.5-coder:7b"
      preferred_models: ["qwen2.5-coder:7b", "llama3.1:8b"]
```

**Model tiers (per-task routing):**
```yaml
llm:
  tiers:
    fast: ["local"]        # classification, importance, compaction
    deep: ["cloud", "local"]  # chat, decompose, tool loop
```

Unset tiers alias the default chain. Unknown names fail closed at startup.

## Adapter Configuration

```yaml
adapters:
  http: { enabled: true, host: "127.0.0.1", port: 19789, cors: true }
  ws:   { enabled: true, port: 19790 }
  mcp:  { enabled: true, port: 19791 }
  grpc: { enabled: true, port: 19792 }
  terminal: { enabled: true, port: 19793 }
```

## Memory Namespaces

```yaml
memory:
  namespaces:
    personal: { residency: any }
    private:  { residency: local_only }  # never leaves your machine
    work:     { residency: any }
```

## Monitoring

```yaml
monitoring:
  services: []                        # external health-check endpoints
  connectivity:
    enabled: true
    interval_secs: 60
    timeout_secs: 5
  power:
    enabled: true
    interval_secs: 60
    defer_maintenance: true
  manifest_health:
    enabled: true
    interval_secs: 120
```
