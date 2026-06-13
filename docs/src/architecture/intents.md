# Intent Taxonomy

Brain exposes 28 intent variants covering all user-facing actions. Intents are classified by the Thalamus using a regex fast-path with LLM fallback.

## Intent Categories

| Category | Intents |
|----------|---------|
| **Inspection** | `Recall`, `MemorySummary`, `SystemStatus`, `ProactivityStatus`, `BudgetStatus`, `List`, `TaskStatus`, `QueryAgents`, `QueryAudit`, `ChannelPreferences` |
| **Memory** | `StoreFact`, `Forget` |
| **Action** | `ExecuteCommand`, `WebSearch`, `SendMessage`, `DelegateTask` |
| **Lifecycle** | `Schedule`, `DecomposeTask`, `Cancel`, `OpenTerminalSession`, `CloseTerminalSession`, `MountMcpServer`, `UnmountMcpServer`, `ReconsentMcpServer` |
| **Governance** | `RespondToApproval`, `ApproveMemoryWriter`, `PruneAudit`, `SetChannelPreference`, `SetProactivity` |
| **Capability** | `ToolCall` |
| **Conversation** | `Chat` |

## Standardized Intent Tokens (SIT)

Every intent can be expressed as a typed `IntentToken` — a JSON object with a `verb` and optional `object`/`modifiers`. This enables programmatic intent dispatch alongside natural language classification.

## Verb Vocabulary

The verb vocabulary is a compile-time constant set, cross-checked by tests that ensure every `Intent` variant resolves through the registry and matches its declared tier hint. Adding a verb requires code changes to the intent enum, auth mapping, handler, and tier hint — so a TOML registry would add friction without flexibility.
