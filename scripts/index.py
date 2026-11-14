from dotenv import load_dotenv
import os
import re
import requests
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from db import db_connection, save_chunks

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
    if not 0 <= overlap < chunk_size:
        raise ValueError("Il faut 0 <= overlap < chunk_size.")

    document_path = Path(document_path)
    text = document_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Documentation vide : {document_path}")

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
        source = f"data_quality_demo/{document_path.name}#{section_id}"
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
        raise ValueError(f"Aucun contenu à indexer : {document_path}")
    return chunks


if __name__ == "__main__":
    chunks = get_chunks_tokens_rag(documentation_path)
    embeddings = embed([chunk["text"] for chunk in chunks])

    with db_connection() as connection:
        # La table vectorielle contient maintenant cette seule documentation.
        # En cas d'échec d'insertion, les anciens chunks sont conservés.
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM chunks")
            save_chunks(connection, chunks, embeddings)

    print(f"{len(chunks)} chunks de {documentation_path.name} enregistrés.")
