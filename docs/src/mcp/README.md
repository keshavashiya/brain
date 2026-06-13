# MCP Integration

Any MCP-compatible client can connect to Brain as a stdio MCP server.

## Configuration

Add to your MCP client config:

```json
{
  "mcpServers": {
    "brain": {
      "command": "brain",
      "args": ["mcp"]
    }
  }
}
```

## Tools exposed via MCP

| Tool | Arguments | Description |
|------|-----------|-------------|
| `memory_search` | `query`, `top_k?`, `namespace?` | Semantic search of Brain memory (facts & episodes) |
| `memory_store` | `subject`, `predicate`, `object` | Store a structured semantic fact |
| `memory_facts` | `subject` | Retrieve all stored facts about a subject |
| `memory_episodes` | `limit?`, `namespace?` | Retrieve recent conversation episodes |
| `user_profile` | — | Retrieve user profile & Brain OS configuration |
| `memory_procedures` | `action`, `trigger?`, `steps?` | Manage learned procedures (list/store/delete) |
| `brain_capabilities` | — | List Brain's live capability manifest |

## MCP Host

Brain can also mount external MCP servers as capabilities. Configure servers in `~/.brain/config.yaml`:

```yaml
mcphost:
  servers:
    - name: "filesystem"
      transport: "stdio"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem"]
```

Mounted servers' tools are registered in the capability manifest alongside native tools, available to every face (CLI, chat, HTTP, etc.).
