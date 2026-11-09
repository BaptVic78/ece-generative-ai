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
    embedding VECTOR(1024) NOT NULL
);
```
The vector dimension must match the embedding model output. This project uses 1024-dimensional embeddings.





