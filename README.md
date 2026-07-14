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

Then type questions and press Enter to interact.

## PDF input

Place your PDF file in the project folder. The application will load and process the PDF automatically.

## Notes

- The first run may take time while the PDF is split and the vector store is built.
- The FAISS index is stored locally so subsequent runs can reuse the embedding data.
- This project uses local models and does not require an external API key.
