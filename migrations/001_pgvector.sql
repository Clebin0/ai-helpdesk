-- AI Helpdesk RAG storage. The default multilingual MiniLM embedding has 384 dimensions.
CREATE TABLE IF NOT EXISTS rag_chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    embedding_model TEXT NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_rag_chunk_source_model UNIQUE (source, chunk_index, embedding_model)
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_model
    ON rag_chunks (embedding_model);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
