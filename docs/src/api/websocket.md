# WebSocket API

Brain exposes a WebSocket API on port **19790** for streaming interactions.

## Connection

```javascript
const ws = new WebSocket('ws://localhost:19790');
```

## Message format

Messages are JSON with the following structure:

```json
{
  "content": "remember my favorite editor is Neovim",
  "namespace": "personal",
  "source": "web-client"
}
```

The server streams responses as JSON-encoded `SignalResponse` messages.
