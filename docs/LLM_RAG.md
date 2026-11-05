# LLM RAG Assistant

This is a separate experimental module beside the original TF-IDF ticket classifier. The classifier remains a traditional ML baseline; the assistant uses an LLM and embeddings to answer questions from indexed support documents.

## Architecture

1. A maintainer submits a document to POST /assistant/ingest.
2. The API splits the text into overlapping chunks and creates embeddings with text-embedding-3-small.
3. Chunk text, source name and vectors are stored in a local SQLite database.
4. A user submits a question to POST /assistant/chat.
5. The question is embedded, chunks are ranked by cosine similarity and low-relevance results are rejected.
6. The Responses API generates an answer using only the retrieved context. The response includes source references such as [S1].

The initial vector search scans the local SQLite collection. This keeps the learning project transparent and easy to run; it is not intended for large multi-tenant corpora. PostgreSQL with pgvector or a dedicated vector database is a future scaling step.

## Models

- Chat: gpt-6-luna by default. Override with OPENAI_CHAT_MODEL.
- Embeddings: text-embedding-3-small by default. Override with OPENAI_EMBEDDING_MODEL.
- Both are configurable. Changing the embedding model requires re-indexing documents because vectors from different models are not comparable.

## Configure locally (PowerShell)

Copy `.env.example` to `.env` and fill in the values locally. The application loads `.env` automatically at startup. Keep `.env` private; it is ignored by Git.

    Copy-Item .env.example .env
    notepad .env

Set `OPENAI_API_KEY` to an API key created in your OpenAI Platform project. Generate two different random values for `ASSISTANT_ACCESS_TOKEN` and `ASSISTANT_ADMIN_TOKEN`; do not reuse the API key. For example, run this twice and use a different output for each token:

    python -c "import secrets; print(secrets.token_urlsafe(32))"

Keep `OPENAI_CHAT_MODEL=gpt-6-luna` and `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` unless you intentionally choose other models available to your API project.

    python -m pip install -r requirements.txt
    python train.py
    uvicorn app:app --reload

Open http://127.0.0.1:8000/docs.

## Index a document

Only submit documentation you are authorized to use. Do not upload personal data, customer records, passwords, access tokens, MFA codes, secrets or unredacted incident logs.

    $body = @{
      source = 'network-dns-runbook.md'
      content = 'DNS troubleshooting: verify affected users, IP address, gateway and configured resolvers. Test name resolution and record the exact error. Escalate if the resolver is unavailable; do not change production settings without approval.'
    } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/assistant/ingest' -Headers @{ 'X-Admin-Token' = $env:ASSISTANT_ADMIN_TOKEN } -ContentType 'application/json' -Body $body

## Ask a question

    $body = @{ question = 'Como começo a diagnosticar uma falha de resolução DNS?'; top_k = 4 } | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/assistant/chat' -Headers @{ 'X-Assistant-Token' = $env:ASSISTANT_ACCESS_TOKEN } -ContentType 'application/json' -Body $body

The response contains the answer, model names, and source metadata. The model is instructed to cite source IDs and say when the retrieved knowledge is insufficient. This reduces unsupported answers but does not guarantee factual correctness; review the cited source before acting.

## Endpoints

| Method | Route | Access |
|---|---|---|
| GET | /assistant/health | Public configuration status; no secrets |
| GET | /assistant/documents | X-Admin-Token |
| POST | /assistant/ingest | X-Admin-Token |
| POST | /assistant/chat | X-Assistant-Token |

Authentication is required for chat and ingestion. If the corresponding token is not configured, the route stays disabled. Keep the API on a trusted network and put it behind TLS and an identity-aware gateway before any shared deployment. Tokens here are a learning-project access gate, not a complete production identity system.

## Current limits and next experiments

- SQLite scans every stored vector; benchmark before scaling.
- Chunking is character-based, not tokenizer-aware.
- Similarity threshold is configurable through RAG_MIN_RELEVANCE and needs evaluation against a curated question set.
- No conversation memory, hybrid search, reranker, streaming or agent tool execution yet.
- There is no automatic action execution. The assistant only retrieves and explains approved documents.
- Add offline evals for retrieval precision/recall, citation correctness, abstention quality, latency and token cost before claiming production readiness.


## Erros de consumo da API

- **HTTP 402:** a conta da OpenAI está sem créditos ou atingiu um limite de gastos/uso. Verifique saldo e limites no painel de cobrança antes de repetir.
- **HTTP 429:** limite temporário de solicitações ou tokens. Aguarde o período indicado pela API e evite novas tentativas em loop.
- **HTTP 502:** falha de comunicação ou erro não classificado do provedor. Consulte os logs locais sem compartilhar chaves ou tokens.

Os erros são apresentados sem expor a chave de API ou o conteúdo interno da resposta do provedor.
