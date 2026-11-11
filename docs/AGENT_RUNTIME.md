# Multi-agent runtime (experimental)

## Endpoint

`POST /agents/assist` reuses the existing `ASSISTANT_ACCESS_TOKEN` and the `X-Assistant-Token` header. Provide a question, a ticket, or both.

Example:

```http
POST /agents/assist
X-Assistant-Token: <ASSISTANT_ACCESS_TOKEN>
Content-Type: application/json
```

```json
{
  "question": "Como diagnosticar falha de DNS?",
  "top_k": 4
}
```

Ticket triage can be requested independently:

```json
{
  "ticket": {
    "title": "Suspeita de phishing",
    "description": "Usuário recebeu um link suspeito.",
    "affected_users": 1,
    "service_stopped": false,
    "vip_or_critical_area": false
  }
}
```

The response includes a trace ID, selected specialists, each specialist result, and an escalation decision.

## Specialists

- **Coordinator** chooses the fixed workflow from the fields supplied by the caller.
- **Triage** reuses the existing deterministic ML classifier and priority/review rules.
- **Knowledge** reuses the existing RAG assistant, local embedding/retrieval stack, and cited answer format.
- **Citation integrity** checks that citation IDs in the answer exist in the retrieved source list.

The citation check is structural only. It does not prove that a claim is semantically supported by a cited passage. Missing or unknown citation IDs require human review.

## Security boundary

This first runtime is intentionally in-process and read-only. It has no shell, browser, filesystem, arbitrary HTTP, direct database-write, ticket-write, or infrastructure-remediation tools. It does not grant OpenClaw access to the host. The existing API token remains required.

The RAG assistant may return an abstention when the retriever has no sufficiently relevant sources. An abstention is surfaced as `status=abstained`; it is not converted into a guessed answer.

## Tests

```powershell
python -m pytest -q tests/test_agent_orchestrator.py tests/test_agent_api.py
```

Run the existing app locally as documented in the repository, then call `POST /agents/assist`.

## Next steps before production

1. Run the full test suite and CI on this branch.
2. Add semantic grounding evaluation against the existing evidence benchmark; do not treat citation-ID validation as a grounding judge.
3. Add caller/tenant identity and rate limits if the API is exposed beyond loopback.
4. Add read-only OpenClaw adapter only after channel identity mapping and tool allowlists are verified.
5. Keep all write actions behind deterministic application policy and human approval.
