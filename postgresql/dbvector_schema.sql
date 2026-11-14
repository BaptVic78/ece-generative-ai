-- Exécuter dans genai_project après installation de pgvector sur le serveur.
-- Crée les tables d'une base neuve et complète une ancienne table chunks.
-- Ne supprime aucune donnée. Avec psql, utiliser -1 pour une transaction unique.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    index_signature TEXT NOT NULL,
    index_settings JSONB NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (content_hash, index_signature)
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    document_id BIGINT REFERENCES documents(id)
);

-- Compatibilité avec les chunks créés avant le suivi des documents.
ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS document_id BIGINT REFERENCES documents(id);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
