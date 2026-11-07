# RAG Tuner: An AI Assistant for Testing and Improving Document Retrieval

## Description

RAG Tuner is a local AI assistant that helps developers choose retrieval settings for their documents. It uses Qwen through Ollama and draws on technical documentation to explain how passage size, the number of retrieved passages, hybrid search, and reranking affect a RAG system. To test its recommendations, the assistant evaluates different configurations on Python documentation using about 100 questions with known relevant passages. It analyzes failed searches, compares retrieval quality and latency, and asks for approval before activating a configuration that improves results on held-out questions within a predefined latency budget. We will compare the assistant with exhaustive search to measure how many experiments each approach needs to reach the same Recall@5.

## Local setup

RAG Tuner uses uv to manage Python dependencies and Ollama to run two models locally. `qwen3.5:4b` generates answers, and `qwen3-embedding:0.6b` converts questions and document passages into vectors for retrieval.

### 1. Install the prerequisites

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

### 4. Verify the local API

```bash
curl -sS http://localhost:11434/api/chat \
  -d '{"model":"qwen3.5:4b","messages":[{"role":"user","content":"Reply with OK."}],"think":false,"stream":false}'
```

The JSON response should contain an answer in `message.content`. Then check the embedding model:

```bash
curl -sS http://localhost:11434/api/embed \
  -d '{"model":"qwen3-embedding:0.6b","input":"What is reranking?"}'
```

The response should contain an `embeddings` array with numeric values. If either command reports `Failed to connect`, launch Ollama from the Windows Start menu and retry.



