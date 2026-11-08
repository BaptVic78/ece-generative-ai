import requests
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"
LLM = "qwen3.5:4b"
urls = ["https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html", 
        "https://docs.haystack.deepset.ai/docs/documentsplitter", 
        "https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html",
        "https://qdrant.tech/documentation/search/hybrid-queries/"
]

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
    response.raise_for_status()
    data = response.json()
    for label, field in [
    ("Chargement de Qwen", "load_duration"),
    ("Lecture du prompt", "prompt_eval_duration"),
    ("Génération de la réponse", "eval_duration"),
    ]:
        print(f"{label} : {data[field] / 1_000_000_000:.1f} s")

    return data["message"]["content"]

def get_chunks_paragraphs(urls):
    chunks = []
    for url in urls : 
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        page = BeautifulSoup(response.content, "html.parser") # page contains the HTML code of the page
        article = page.select_one("article, main, [role='main']") # article contains the 1st HTML element that correspond to 1 of the 3 selectors
        if article is None:
            raise ValueError("Impossible de trouver le contenu principal de la page.")

        for element in article.find_all("p") :
            text = element.get_text(" ", strip=True)
            if text:
                chunks.append({
                    "text": text, 
                    "url": url
                })
    return chunks

def get_chunks_tokens(urls, chunk_size=200, overlap=50):
    chunks = []
    for url in urls:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        page = BeautifulSoup(response.content, "html.parser")
        article = page.select_one("article, main, [role='main']")
        if article is None:
            raise ValueError("Impossible de trouver le contenu principal de la page.")

        paragraphs = []
        for element in article.find_all(["p", "li"]):
            text = element.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)

        text_page = "\n\n".join(paragraphs)
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

def retrieve(embeddings, chunks, question, k=4):
    q = embed([question])[0]
    scores = embeddings @ q / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q)) # cosine similarity formula
    return [chunks[i] for i in np.argsort(scores)[::-1][:k]] # returns top 3 similary chunks

def answer_question(question, chunks, embeddings):
    passages = retrieve(embeddings, chunks, question, k=4)
    context = "\n\n".join(
        f"[{i}] Source: {passage['url']}\n{passage['text']}"
        for i, passage in enumerate(passages, start=1)
    )

    prompt = f"""Answer in the language of the question using only the passages below.
Cite the supporting passage's exact URL after each factual claim.
If the passages do not support an answer, JUST SAY you do not know.

Passages:
{context}

Question:
{question}
"""
    return ask_llm(prompt), passages

def run_llm_with_rag(chunks, embeddings):
    while True:
        question = input("Ask: ")
        answer, passages = answer_question(question, chunks, embeddings)
        print(answer)
        for i, passage in enumerate(passages, start=1):
            print(f"[{i}] {passage['url']}")
    
if __name__ == "__main__":
    chunks = get_chunks_tokens(urls)
    embeddings = embed([chunk["text"] for chunk in chunks])
    run_llm_with_rag(chunks, embeddings)
    # question = (
    # "Après le découpage d’un document, quel champ permet de retrouver "
    # "le document d’origine dans Haystack ?"
    # )

    # passages = retrieve(embeddings, chunks, question, k=10)

    # for rang, passage in enumerate(passages, start=1):
    #     if "source_id" in passage["text"]:
    #         print(f"Passage contenant source_id trouvé au rang {rang}")
    #         print(passage["text"])
    #         break
    # else:
    #     print("Aucun passage contenant source_id dans le top 10.")




