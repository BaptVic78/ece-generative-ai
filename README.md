# RAG Tuner: An AI Assistant for Testing and Improving Document Retrieval

## Description

RAG Tuner is a local AI assistant that helps developers choose retrieval settings for their documents. It uses Qwen through Ollama and draws on technical documentation to explain how passage size, the number of retrieved passages, hybrid search, and reranking affect a RAG system. To test its recommendations, the assistant evaluates different configurations on Python documentation using about 100 questions with known relevant passages. It analyzes failed searches, compares retrieval quality and latency, and asks for approval before activating a configuration that improves results on held-out questions within a predefined latency budget. We will compare the assistant with exhaustive search to measure how many experiments each approach needs to reach the same Recall@5.

## Install the prerequisites

RAG Tuner uses uv to manage Python dependencies and Ollama to run two models locally. `qwen3.5:4b` generates answers, and `qwen3-embedding:0.6b` converts questions and document passages into vectors for retrieval.

### 1. Install uv and ollama

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Ollama for Windows](https://ollama.com/download/windows). Launch Ollama, open a new Git Bash terminal, and check that both commands are available:

```bash
uv --version
ollama --version
```

### 2. Set up the Python environment

From the root of the cloned repository, run:
```bash
uv sync --locked
```

This creates the `.venv` environment and installs the dependency versions recorded in `uv.lock`. To run Python inside this environment, use `uv run`:

```bash
uv run python --version
```

### 3. Download the models

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

### 4. PostgreSQL and pgvector

We use PostgreSQL with the pgvector extension to store chunks, their sources, and their embeddings.

1. Install [PostgreSQL](https://www.postgresql.org/download/).
2. Install pgvector by following the [official installation instructions](https://github.com/pgvector/pgvector#installation).
3. Open pgAdmin and connect to your PostgreSQL server using the host, port, username, and password configured during installation.

Open the Query Tool on the existing `postgres` database and execute this command separately:
```sql
CREATE DATABASE genai_project;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    corpus TEXT NOT NULL 
);
```
The vector dimension must match the embedding model output. This project uses 1024-dimensional embeddings.

## Run the project

Run the following commands from the repository root.

### Index the documents

```bash
uv run scripts/index.py rag
uv run scripts/index.py python
```

These scripts extract the RAG documentation, split it into chunks, generate
embeddings with Ollama, and store the chunks, sources, and embeddings in
PostgreSQL. Existing entries for the indexed sources are replaced.
Approximately 400 chunks for the RAG documentation, and 5337 for the python documentation.

Run this step before the first use and whenever the documents, chunking
configuration, or embedding model change.

> [!WARNING]
> Without GPU, the script takes a lot of time to run.

### Chat with the assistant

```bash
uv run scripts/rag.py
```

The assistant searches the indexed documents to answer your questions.
Enter `quit` or `exit` to stop.

### Evaluate the RAG

```bash
uv run scripts/eval.py
```

This script runs the questions from `tests_rag/test.jsonl` against the
indexed documents. It saves a timestamped results file in `scripts/`:

```text
resultats-YYYYMMDD-HHMMSS-ffffff.jsonl
```

Each record contains the question, expected answer, generated answer,
retrieved passages, and response time. Correctness and citation fields
are left empty for manual assessment.


## Current limitations

The RAG assistant can retrieve relevant passages, but some answers still
misinterpret them or add unsupported details.

Citations are frequently present, but their presence does not guarantee
that they support the answer.

The assistant generally abstains when local experimental results are
unavailable. However, some abstentions include unnecessary or inaccurate
claims.

Evaluation results are saved as JSONL files. Answer correctness and
citation validity are assessed manually.



