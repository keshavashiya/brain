# Security

Brain's security model is built on layered guarantees.

## Authentication

Every API request requires an API key (generated at `brain init`). Keys can be scoped to specific agents with limited permissions.

## Authorization tiers

All capabilities are tagged with a safety tier:

| Tier | Examples | Requires confirmation? |
|------|----------|----------------------|
| **Read** | Memory search, status, audit query | No |
| **Write** | Store fact, set preference | No |
| **Execute** | Run command, web search | Yes (nonce-based) |
| **Destructive** | Delete memory, prune audit | Yes + budget check |
| **External** | Send message, delegate task | Yes + cost check |

## Confirmation engine

Destructive and external actions require a nonce-based approval flow. The engine supports:
- Standing approvals (with optional TTL and scope)
- Confirmation timeouts (pauses when user is away)
- Cross-channel confirmation correlation

## Audit trail

Every action is recorded in an append-only SQLite audit trail with immutable triggers. The audit covers who did what, when, and the authorization decision.

## Sandbox

Command execution runs in a sandbox with:
- Process-group SIGKILL on timeout
- Binary allowlist
- rlimits (CPU, address space, file count, file size)
- macOS `sandbox-exec` / Linux `unshare` network isolation

## Data residency

Namespaces can be marked `local_only`, preventing their data from reaching any non-local LLM provider. Enforcement happens at every egress point — recall, embedding, export.

## Credential vault

Secrets are stored in the OS-native keychain (macOS Keychain, Linux Secret Service) with an encrypted-file fallback (Argon2id + AES-256-GCM).
