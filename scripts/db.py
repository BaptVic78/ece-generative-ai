from dotenv import load_dotenv
import os
from pathlib import Path
import psycopg
from pgvector.psycopg import register_vector

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

def db_connection():
    return psycopg.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
    )

def ensure_document_schema(connection):
    """Ajouter le suivi des documents sans supprimer les chunks existants."""
    schema_path = Path(__file__).resolve().parent.parent / "postgresql" / "dbvector_schema.sql"
    with connection.transaction():
        with connection.cursor() as cursor:
            # Sérialiser la migration lors de premiers dépôts simultanés.
            cursor.execute("SELECT pg_advisory_xact_lock(2026111401)")
            cursor.execute(
                """
                SELECT to_regclass('documents') IS NOT NULL,
                       EXISTS (
                           SELECT 1 FROM pg_attribute
                           WHERE attrelid = to_regclass('chunks')
                             AND attname = 'document_id'
                             AND NOT attisdropped
                       )
                """
            )
            has_documents, has_document_id = cursor.fetchone()
            if not (has_documents and has_document_id):
                cursor.execute(schema_path.read_text(encoding="utf-8"))


def save_chunks(connection, chunks:list, embeddings, replace_sources=False, document_id=None):
    if len(chunks) != len(embeddings):
        raise ValueError("Chaque chunk doit avoir un embedding.")
    if document_id is None:
        ensure_document_schema(connection)
    register_vector(connection)
    query = """
        INSERT INTO chunks (text, source, embedding, document_id)
        VALUES (%s, %s, %s, %s);
    """
    # Un échec annule le remplacement et les insertions de ce lot.
    with connection.transaction():
        with connection.cursor() as cur:
            if replace_sources:
                sources = sorted({chunk["url"] for chunk in chunks})
                cur.execute("DELETE FROM chunks WHERE source = ANY(%s)", (sources,))
            for i, chunk in enumerate(chunks):
                cur.execute(
                    query,
                    (chunk["text"], chunk["url"], embeddings[i], document_id),
                )


def retrieve(connection, question_embedding, k):
    register_vector(connection)
    with connection.cursor() as cur:
        cur.execute(
            """
            SELECT text, source
            FROM (
                SELECT DISTINCT ON (source, text)
                    text, source, embedding <=> %s AS distance
                FROM chunks
                ORDER BY source, text, distance
            ) AS unique_chunks
            ORDER BY distance
            LIMIT %s
            """,
            (question_embedding, k),
        )   
        results = cur.fetchall()
    return [
        {"text": text, "url": source}
        for text, source in results
    ]
