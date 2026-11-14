from dotenv import load_dotenv
import os
import hashlib
import json
import re
import requests
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from db import db_connection, ensure_document_schema, save_chunks

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

OLLAMA = os.environ.get("OLLAMA_URL")
EMBED_MODEL = os.environ.get("EMBED_MODEL")

tokenizer = None

documentation_path = (
    Path(__file__).resolve().parent.parent
    / "data_quality_demo"
    / "01_doc_metier.md"
)

def embed(texts, batch_size=16):
    embeddings = []

    for debut in range(0, len(texts), batch_size):
        lot = texts[debut:debut + batch_size]
        print(
            f"Embeddings : {debut + 1} à {debut + len(lot)} / {len(texts)}",
            flush=True,
        )

        response = requests.post(
            f"{OLLAMA}/api/embed",
            json={"model": EMBED_MODEL, "input": lot},
            timeout=(10, 300),
        )
        response.raise_for_status()
        embeddings.extend(response.json()["embeddings"])

    return np.array(embeddings)

def get_chunks_tokens_rag(document_path, chunk_size=350, overlap=80):
    document_path = Path(document_path)
    return get_markdown_chunks(
        document_path.read_text(encoding="utf-8-sig"),
        f"data_quality_demo/{document_path.name}",
        chunk_size,
        overlap,
    )


def get_markdown_chunks(text, source_name, chunk_size=350, overlap=80):
    """Découper le texte Markdown d'un fichier local ou déposé dans Streamlit."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Il faut 0 <= overlap < chunk_size.")
    if not text.strip():
        raise ValueError("Le document Markdown est vide.")

    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_MODEL"])

    # Une section ne traverse pas le titre de la suivante.
    sections = []
    title = "Introduction"
    lines = []
    fence = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
        if fence is None and re.match(r"^#{1,6} ", line):
            if "\n".join(lines).strip():
                sections.append((title, "\n".join(lines).strip()))
            title = line.lstrip("#").strip()
            lines = []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((title, "\n".join(lines).strip()))

    chunks = []
    for number, (title, body) in enumerate(sections, start=1):
        # Répéter le titre dans chaque fragment pour conserver le code métier.
        prefix_tokens = tokenizer.encode(title + "\n\n", add_special_tokens=False)
        body_size = chunk_size - len(prefix_tokens)
        if body_size <= overlap:
            raise ValueError(f"Titre trop long pour ce découpage : {title}")

        code = re.search(r"\b(?:DQ-[A-Z]+[0-9]+|[A-Z]{2,4}-[0-9]+)\b", title)
        section_id = code.group(0) if code else f"section-{number}"
        source = f"{source_name}#{section_id}"
        tokens = tokenizer.encode(body, add_special_tokens=False)

        for start in range(0, len(tokens), body_size - overlap):
            end = start + body_size
            chunks.append({
                "text": tokenizer.decode(prefix_tokens + tokens[start:end]),
                "url": source,
            })
            if end >= len(tokens):
                break

    if not chunks:
        raise ValueError(f"Aucun contenu à indexer : {source_name}")
    return chunks


def index_markdown(connection, filename, content, chunk_size=350, overlap=80,
                   source_name=None, reuse_existing=False):
    """Indexer un nouveau contenu, ou réutiliser son index déjà enregistré."""
    filename = Path(filename.replace("\\", "/")).name
    if not filename.lower().endswith(".md"):
        raise ValueError("Seuls les fichiers .md sont acceptés.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("Il faut 0 <= overlap < chunk_size.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Le fichier Markdown doit être encodé en UTF-8.") from error
    if not text.strip():
        raise ValueError("Le document Markdown est vide.")

    content_hash = hashlib.sha256(content).hexdigest()
    settings = {
        "embedding_model": EMBED_MODEL,
        "tokenizer_model": os.environ["TOKENIZER_MODEL"],
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunker_version": 1,
    }
    index_signature = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()
    source = source_name or f"{filename}@{content_hash[:12]}"

    ensure_document_schema(connection)
    # Le verrou couvre aussi les embeddings : deux dépôts identiques ne les
    # recalculent pas simultanément. Toute erreur annule document et chunks.
    lock_key = int(hashlib.sha256(
        (content_hash + index_signature).encode("ascii")
    ).hexdigest()[:15], 16)

    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
            cursor.execute(
                """
                SELECT d.id, d.filename, d.source, COUNT(c.id)
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.content_hash = %s AND d.index_signature = %s
                GROUP BY d.id
                """,
                (content_hash, index_signature),
            )
            existing = cursor.fetchone()
            if existing and existing[3] > 0:
                return {
                    "document_id": existing[0],
                    "filename": existing[1],
                    "source": existing[2],
                    "chunk_count": existing[3],
                    "already_indexed": True,
                }

            if existing:
                source = existing[2]
            chunks = get_markdown_chunks(text, source, chunk_size, overlap)
            legacy_ids = []
            if reuse_existing:
                # Migration explicite : vérifier tous les textes et leurs sources.
                # Le modèle et les réglages doivent être ceux de l'ancien index.
                cursor.execute(
                    """
                    SELECT id, text, source FROM chunks
                    WHERE document_id IS NULL AND source = ANY(%s)
                    FOR UPDATE
                    """,
                    (sorted({chunk["url"] for chunk in chunks}),),
                )
                legacy_rows = cursor.fetchall()
                expected = {(chunk["text"], chunk["url"]) for chunk in chunks}
                actual = {(row[1], row[2]) for row in legacy_rows}
                if len(legacy_rows) == len(chunks) and actual == expected:
                    legacy_ids = [row[0] for row in legacy_rows]

            if not legacy_ids:
                embeddings = embed([chunk["text"] for chunk in chunks])
                if len(chunks) != len(embeddings):
                    raise ValueError("Chaque chunk doit avoir un embedding.")

            # Reconstituer aussi un index dont les chunks ont été supprimés.
            if existing:
                document_id = existing[0]
            else:
                cursor.execute(
                    """
                    INSERT INTO documents (
                        filename, source, content, content_hash,
                        index_signature, index_settings
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (filename, source, text, content_hash, index_signature,
                     json.dumps(settings)),
                )
                document_id = cursor.fetchone()[0]

            if legacy_ids:
                cursor.execute(
                    "UPDATE chunks SET document_id = %s WHERE id = ANY(%s)",
                    (document_id, legacy_ids),
                )
            else:
                save_chunks(connection, chunks, embeddings, document_id=document_id)

    return {
        "document_id": document_id,
        "filename": filename,
        "source": source,
        "chunk_count": len(chunks),
        "already_indexed": False,
    }


if __name__ == "__main__":
    with db_connection() as connection:
        result = index_markdown(
            connection,
            documentation_path.name,
            documentation_path.read_bytes(),
            source_name=f"data_quality_demo/{documentation_path.name}",
            reuse_existing=True,
        )

    action = "déjà indexés" if result["already_indexed"] else "enregistrés"
    print(f"{result['chunk_count']} chunks de {result['filename']} {action}.")
