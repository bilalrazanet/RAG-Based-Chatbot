# RAG-Based Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, and FAISS.
This project ingests a PDF document, splits it into chunks, embeds the text, builds a FAISS vector store,
and serves a chat interface that answers questions using document-based retrieval.

## Key Features

- PDF document ingestion and parsing
- Text chunking with context overlap
- Semantic retrieval via FAISS vector search
- Local embeddings with Sentence Transformers
- LLM generation with Google Flan-T5
- Browser and terminal chatbot modes
- No external API key required

## Setup

1. Install Python 3.11.
2. Create and activate the virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scriptsctivate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Browser chatbot

```bash
.venv\Scripts\python.exe web_chatbot.py
```
Open your browser at:

```bash
http://127.0.0.1:8000
```

### Terminal chatbot

```bash
.venv\Scripts\python.exe rag_chatbot.py
```

### Model flow

1. Load the PDF file from the project folder.
2. Split text into chunks.
3. Create semantic embeddings.
4. Store vectors in FAISS.
5. Retrieve relevant chunks for each query.
6. Generate answers with the LLM.
