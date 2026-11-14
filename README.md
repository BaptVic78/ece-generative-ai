# RAG Tuner: An AI Assistant for Testing and Improving Document Retrieval

## Description

The current version is a local RAG assistant that answers questions using technical documentation, with Qwen through Ollama, PostgreSQL/pgvector retrieval, and a Cross-Encoder reranker. Evaluation records retrieved passages, Recall@k, generated answers, token usage, and latency.

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
```

Open a new Query Tool connected to `genai_project`, then open and execute
`postgresql/dbvector_schema.sql`. This file creates the pgvector extension,
the `documents` table, the `chunks` table and their foreign key.

Alternatively, from the repository root, use psql (adapt the port if needed):

```bash
psql -h 127.0.0.1 -p 5433 -U postgres -d genai_project -1 -f postgresql/dbvector_schema.sql
```

The same script also adds `document_id` to an existing `chunks` table.
It preserves existing rows; it does not populate their document references.

The vector dimension must match the embedding model output. This project uses 1024-dimensional embeddings.

## Run the project

Run the following commands from the repository root.

### Index the documents

```bash
uv run scripts/index.py
```

This script indexes the local Markdown file
`data_quality_demo/01_doc_metier.md`. Each document is identified by its
SHA-256 content hash and indexing settings (embedding model, tokenizer,
chunk size and overlap). An already indexed document is reused without
recomputing its embeddings, even if uploaded under a different filename.

New documents are added without deleting existing chunks. A changed document
is stored as a new version alongside previous documents. The application
currently searches all indexed documents together.

The first indexing operation automatically creates the `documents` table
and adds a nullable `document_id` foreign key to `chunks`. Existing chunks
are preserved. Documents indexed before this migration have no recorded
fingerprint and will be registered on their next import.

> [!WARNING]
> Without GPU, the script takes a lot of time to run.

### Upload Markdown documents in Streamlit

```bash
uv run streamlit run app.py
```

Attach one or more UTF-8 Markdown files directly in the chat input and send
them, with or without a question. Files are indexed automatically before
the question is answered. A previously indexed document is reused.
The chat searches the vector database and displays the passages used.

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

This script runs the questions from `tests/rag/test.jsonl` against the
indexed documents. It saves a timestamped results file in `tests/rag/`:

```text
resultats-rag-YYYYMMDD-HHMMSS-ffffff.jsonl
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



