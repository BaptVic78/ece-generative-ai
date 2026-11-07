import requests
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"
LLM = "qwen3.5:4b"
URL = "https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html"

def embed(texts):
    response = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    return np.array(response.json()["embeddings"])

def ask_llm(prompt):
    response = requests.post(f"{OLLAMA}/api/chat", json={
        "model": LLM, 
        "messages": [{"role":"user", "content": prompt}],
        "think": False,
        "stream": False
    })
    return response.json()["message"]["content"]

def get_chunks(url):
    chunks = []
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    page = BeautifulSoup(response.content, "html.parser")
    article = page.select_one("article, main, [role='main']")
    if article is None:
        raise ValueError("Impossible de trouver le contenu principal de la page.")

    for element in article.find_all("p") :
        text = element.get_text(" ", strip=True)
        if text:
            chunks.append(text)
    return chunks

def retrieve(embeddings, chunks, question, k=3):
    q = embed([question])[0]
    scores = embeddings @ q / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q)) # cosine similarity formula
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]] # returns top 3 similary chunks

chunks = get_chunks(URL)
print(chunks)
embeddings = embed(chunks)
print(embeddings)

while True:
    question = input("Ask: ")
    context = "\n\n".join(retrieve(embeddings=embeddings, chunks=chunks, question=question, k=3))
    prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}"""
    print("\n" + ask_llm(prompt) + "\n")


