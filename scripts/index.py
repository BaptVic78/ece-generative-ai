from dotenv import load_dotenv
import os
import requests
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from db import db_connection, save_chunks

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

OLLAMA = os.environ.get("OLLAMA_URL")
EMBED_MODEL = os.environ.get("EMBED_MODEL")

tokenizer = None

urls_path = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "rag_urls.txt"
)

urls = [
    line.strip()
    for line in urls_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

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

def get_chunks_tokens(urls, chunk_size=350, overlap=80):
    if not 0 <= overlap < chunk_size:
        raise ValueError("Il faut 0 <= overlap < chunk_size.")
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER_MODEL"])

    chunks = []
    for url in urls:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        page = BeautifulSoup(response.content, "html.parser")

        if url.startswith("https://arxiv.org/abs/"):
            abstract = page.select_one("blockquote.abstract")
            if abstract is None:
                raise ValueError(f"Résumé arXiv introuvable : {url}")

            text_page = abstract.get_text(" ", strip=True)
        else:
            article = page.select_one("article, main, [role='main']")
            if article is None:
                raise ValueError(f"Contenu principal introuvable : {url}")

            # Retirer la navigation et garder les blocs de code/tableaux entiers.
            for element in article.select("nav, aside, script, style"):
                element.decompose()
            paragraphs = []
            # Les signatures et descriptions d'API Sphinx utilisent dt et dd.
            for element in article.find_all(
                ["h1", "h2", "h3", "h4", "p", "li", "pre", "table", "dt", "dd"]
            ):
                if element.find_parent(["li", "pre", "table", "dt", "dd"]) is not None:
                    continue
                if element.name == "pre":
                    # Préserver les espaces du code, sans séparer chaque span HTML.
                    text = element.get_text()
                elif element.name == "table":
                    rows = []
                    for row in element.find_all("tr"):
                        cells = row.find_all(["th", "td"], recursive=False)
                        rows.append(" | ".join(cell.get_text(" ", strip=True) for cell in cells))
                    text = "\n".join(rows)
                else:
                    text = element.get_text(" ", strip=True)
                paragraphs.append(text)
            text_page = "\n\n".join(text for text in paragraphs if text)
        tokens = tokenizer.encode(text_page, add_special_tokens=False)

        for debut in range(0, len(tokens), chunk_size - overlap):
            fin = debut + chunk_size  # which tokens we take   
            chunks.append({
                "text": tokenizer.decode(tokens[debut:fin]), 
                "url": url
            })
            if fin >= len(tokens):
                break
        
    return chunks

if __name__ == "__main__":
    chunks = get_chunks_tokens(urls)
    embeddings = embed([chunk["text"] for chunk in chunks])
    with db_connection() as connection:
        save_chunks(connection, chunks, embeddings, replace_sources=True)
    print(f"{len(chunks)} chunks enregistrés.")

