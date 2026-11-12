# RzChatbot

A local, tool-using chatbot built with **LangGraph**, **Ollama**, and **Streamlit**. It streams replies, keeps persistent per-thread memory, searches the web through **SearXNG**, does math accurately, and answers questions about documents you upload using a hybrid **RAG** pipeline.

## Features

- **Streaming chat UI** with per-conversation history (Streamlit).
- **Persistent memory** via LangGraph's SQLite checkpointer and a conversations table.
- **Document Q&A (RAG)**: upload PDF / DOCX / TXT / Markdown files per chat and ask questions about them.
- **Hybrid retrieval**: BM25 + dense vector search (MMR), followed by a cross-encoder reranker.
- **Tool calling**: the agent decides when to use document search, web search, or the calculator.
- **Source citations**: answers from documents include the file name and page.
- **Auto-generated chat titles** from the first message.
- **Docker support** with model weights baked into the image.

## Architecture

```
Streamlit UI (stream_streamlit_frontend.py)
        |
        v
LangGraph StateGraph (backend.py)
   chat_node  <--->  tools node
        |                |
        |                +-- search_documents      (rag.py)
        |                +-- list_indexed_documents (rag.py)
        |                +-- search_web            (search_client.py -> SearXNG)
        |                +-- calculator            (utility.py)
        |
        +-- ChatOllama  (qwen3.5-9b-abliterated)
        +-- SqliteSaver (chatbot.db)
```

Documents are chunked, embedded with `nomic-embed-text` through Ollama, and stored in a per-thread Chroma collection under `.data/chroma`. At query time the retriever fuses BM25 and dense results, reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and returns the top passages with citations.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Orchestration | LangGraph |
| LLM runtime | Ollama (`ChatOllama`) |
| Embeddings | `nomic-embed-text` (Ollama) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| Vector store | Chroma |
| Retrieval | BM25 + MMR dense, reciprocal fusion, cross-encoder rerank |
| UI | Streamlit |
| Persistence | SQLite (conversations + LangGraph checkpoints) |
| Web search | SearXNG |

## Project Structure

```
.
├── backend.py                    # LangGraph graph, model, system prompt
├── tools.py                      # Tools exposed to the agent
├── rag.py                        # Ingestion + hybrid retrieval
├── search_client.py              # SearXNG client
├── utility.py                    # Exact calculator
├── database.py                   # SQLite conversation store
├── stream_streamlit_frontend.py  # Streamlit app
├── download_models.py            # Pre-download reranker weights
├── requirements.txt
├── Dockerfile
├── .env.example
└── .data/                        # Runtime data (gitignored)
    ├── chroma/                   # Vector collections
    ├── uploads/                  # Uploaded files
    └── models/                   # HuggingFace model cache
```

## Prerequisites

- **Python 3.10+**
- **Ollama** running locally with:
  - a chat model (default: `dmtx/qwen3.5-9b-abliterated:latest`)
  - `nomic-embed-text` for embeddings
- **SearXNG** instance with the JSON API enabled (for web search)

Pull the models:

```bash
ollama pull dmtx/qwen3.5-9b-abliterated:latest
ollama pull nomic-embed-text
```

## Setup

```bash
git clone git@github.com:Rahulrajln1111/RzChatbot.git
cd RzChatbot

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env if your SearXNG URL is different
```

If you use a different chat or embedding model, update the model names in `backend.py` and `rag.py`.

## Run

```bash
streamlit run stream_streamlit_frontend.py
```

Open http://localhost:8501.

### Using documents

1. Open or start a chat.
2. Expand **Documents in this chat** and upload one or more files.
3. Click **Index documents**.
4. Ask questions normally. The agent checks the indexed files first and cites the source and page.

Each conversation has its own Chroma collection, so documents do not leak between chats.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARXNG_URL` | required | Base URL of the SearXNG instance |
| `MODEL_CACHE_DIR` | `.data/models` | Where the reranker is cached |
| `RERANK_ENABLED` | `1` | Set to `0` to skip the cross-encoder |
| `HF_HOME` | `.data/models` | HuggingFace cache directory |
| `HF_HUB_OFFLINE` | unset | Set to `1` to run fully offline after pre-download |
| `OLLAMA_HOST` | Ollama default | Point the app at a remote Ollama server |

## Docker

The image pre-downloads the reranker so containers do not fetch weights at runtime.

```bash
docker build -t rzchatbot .

docker run --rm -p 8501:8501 \
  -e SEARXNG_URL=http://host.docker.internal:8081 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  rzchatbot
```

Ollama and SearXNG must be reachable from the container (or run as separate services).

## Notes and Limitations

- The default model and embeddings are pulled by Ollama; this project does not host them.
- The cross-encoder improves relevance but adds startup and CPU cost. Disable it with `RERANK_ENABLED=0` on constrained hosts.
- Web search requires SearXNG with `format=json` enabled.
- `chatbot.db` and `.data/` are local state and are gitignored.

## License

No license has been specified for this project.
