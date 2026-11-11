from dotenv import load_dotenv
import os
from pathlib import Path
import psycopg
from pgvector.psycopg import register_vector
import numpy as np

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

def save_chunks(connection, chunks:list, embeddings, replace_sources=False):
    if len(chunks) != len(embeddings):
        raise ValueError("Chaque chunk doit avoir un embedding.")
    register_vector(connection)
    query = """
        INSERT INTO chunks (text, source, embedding)
        VALUES (%s, %s, %s);
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
                    (chunk["text"], chunk["url"], embeddings[i]),
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
