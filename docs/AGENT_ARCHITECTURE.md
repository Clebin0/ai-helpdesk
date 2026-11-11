# Multi-agent architecture

## Scope

This document defines a safe, incremental agent layer around the existing Helpdesk Triage and experimental LLM/RAG Assistant. The existing classifier, RAG retrieval, citation generation, and HTTP contracts remain authoritative. This phase is design/documentation only: it does not enable autonomous ticket or infrastructure actions.

## Roles

- **Coordinator**: routes a request to a specialist and composes the final response. It cannot call external systems directly.
- **Triage specialist**: consumes the existing triage API output and explains category, priority, and escalation signals. It cannot override deterministic policy.
- **Knowledge specialist**: queries the existing RAG assistant and must retain its source citations. It must abstain when sources are insufficient.
- **Grounding reviewer**: checks whether each material claim is supported by retrieved sources and flags unsupported claims; it cannot fabricate citations.
- **Escalation analyst**: identifies security, critical-impact, low-confidence, and ambiguous cases for human review.

## Request flow

1. Authenticate the caller and resolve tenant/project scope.
2. Run deterministic validation and the existing triage classifier.
3. Route knowledge questions to the existing RAG API.
4. Run grounding review over the answer and citations.
5. Return the answer, evidence, uncertainty, and human-review reason.
6. Record minimal audit metadata without copying sensitive ticket text into shared agent memory.

## Tool policy

Allowed initially: existing local read-only health, triage, RAG query, metrics, and evaluation endpoints after authentication is implemented for the caller. Tool schemas must validate input and cap payload sizes.

Denied: ticket updates, replies, deletes, shell execution, arbitrary HTTP, filesystem access, infrastructure changes, credential reads, and direct database writes.

## Model policy

Use the local Ollama model for low-risk classification/formatting experiments. Small or quantized models must not be trusted to enforce authorization or resist prompt injection. Authorization is enforced in application code, not in prompts. Keep a deterministic fallback and a human escalation path.

## Acceptance criteria

- Existing API and classifier behavior remain unchanged.
- RAG citations are preserved end-to-end.
- Unsupported or conflicting claims trigger abstention/escalation.
- Tenant and user authorization are checked server-side.
- Tool calls are allowlisted, read-only, bounded, logged, and tested.
- No secret or raw sensitive ticket text is stored in agent memory.
- Unit, integration, and regression tests pass before any runtime integration.

## Rollout

Phase 1: role contracts and architecture (this document).
Phase 2: typed internal interfaces and mocked specialist tests.
Phase 3: authenticated read-only runtime adapters.
Phase 4: offline evaluation against the existing benchmark.
Phase 5: optional OpenClaw channel adapter, isolated from business logic.
