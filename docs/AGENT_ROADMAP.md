# Agent roadmap

1. **Contracts** — define typed input/output schemas, trace IDs, timeouts, token budgets, and explicit abstention states.
2. **Read-only specialists** — implement triage, retrieval, grounding, and escalation adapters around current endpoints; no write tools.
3. **Evaluation** — add multi-agent test cases for answerable/unanswerable questions, prompt injection in retrieved documents, citation preservation, and tool denial.
4. **Observability** — log agent role, tool name, duration, outcome, and policy decision; redact user content and secrets.
5. **Channel integration** — expose a narrow authenticated adapter to OpenClaw only after API-level authorization is verified.

Do not merge experimental retrieval modes into the production chat path without benchmark evidence. Do not use an LLM as the authorization layer.
