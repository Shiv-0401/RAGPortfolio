# RAGPortfolio

A Retrieval Augmented Generation (RAG) chatbot that answers questions about Shiv Patel's projects, resume, and experience, grounded in his actual GitHub READMEs and resume.

## What is RAG?

Retrieval Augmented Generation combines a retrieval step with a generation step:

1. **Retrieve** — when a question comes in, the system searches a vector database of document chunks for the pieces of text most semantically similar to the question.
2. **Augment** — those retrieved chunks are inserted into a prompt as context.
3. **Generate** — a large language model reads the question plus the retrieved context and writes an answer grounded in that context, rather than relying purely on what it memorized during training.

This lets a general-purpose LLM answer accurately and specifically about a narrow, private knowledge base (in this case, one person's resume and project docs) without needing to be fine-tuned.

## How this system works

1. **Ingestion** ([src/ingest.py](src/ingest.py)) — loads every `.txt`, `.md`, and `.pdf` file in `data/documents/`, splits them into ~500-character chunks with 50-character overlap (`RecursiveCharacterTextSplitter`), embeds each chunk locally with `sentence-transformers/all-MiniLM-L6-v2`, and stores the vectors in a persistent ChromaDB collection at `data/chroma_db/`.
2. **Retrieval** ([src/retriever.py](src/retriever.py)) — embeds an incoming question with the same model and queries ChromaDB for the top 5 most relevant chunks, along with their source document name and relevance score.
3. **Generation** ([src/chain.py](src/chain.py)) — a LangChain pipeline stuffs the retrieved chunks into a prompt template and sends it to Groq's `llama-3.1-8b-instant` model, which generates an answer. The source documents used are returned alongside the answer.
4. **Interface** ([app/app.py](app/app.py), [app/templates/index.html](app/templates/index.html)) — a small Flask app exposes a dark-themed chat UI (`GET /`), a JSON chat endpoint (`POST /chat`), and a health check (`GET /health`).

## Tech stack

- **LangChain** — orchestrates the retriever → prompt → LLM pipeline
- **ChromaDB** — persistent local vector database
- **sentence-transformers (all-MiniLM-L6-v2)** — free, local embedding model
- **Groq API (openai/gpt-oss-20b)** — fast, free-tier LLM inference (llama-3.1-8b-instant was originally used but has since been deprecated by Groq)
- **Flask** — web server and chat UI
- **Bootstrap 5** — dark-themed, mobile-responsive chat interface

## Setup

1. **Create a virtual environment and install dependencies:**

   ```bash
   /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set your Groq API key:**

   ```bash
   cp .env.example .env
   # then edit .env and set GROQ_API_KEY=<your key from console.groq.com>
   ```

3. **Add source documents** (already includes sample `resume.txt`, `projects.md`, and `skills.md`) to `data/documents/`.

## Running

1. **Index the documents into ChromaDB:**

   ```bash
   /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 src/ingest.py
   ```

   This prints how many chunks were indexed.

2. **Start the web app:**

   ```bash
   /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 app/app.py
   ```

   Visit [http://localhost:5004](http://localhost:5004) to chat.

3. **Re-index** any time you add or change documents in `data/documents/` by re-running `src/ingest.py`.

## API

- `GET /` — chat interface
- `POST /chat` — body `{"question": "..."}`, returns `{"answer": "...", "sources": ["resume.txt", ...]}`
- `GET /health` — returns `{"status": "ok", "indexed_documents": <count>}`
