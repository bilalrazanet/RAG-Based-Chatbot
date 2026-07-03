# Project Index - PDF RAG Chatbot

## Project Overview
A local retrieval-augmented generation (RAG) chatbot built with LangChain that answers questions based on a PDF document. Uses embeddings for semantic search and a language model for answer generation.

---

## Project Structure

```
d:\Rag practice\
├── .venv/                          # Python virtual environment
├── __pycache__/                    # Python cache files
├── faiss_store/                    # Vector database storage
│   └── index.faiss                 # FAISS vector index
├── 2506.18027v3.pdf               # Source document (research paper)
├── rag_chatbot.py                 # Main chatbot application
├── web_chatbot.py                 # Browser-based chatbot server
├── download_models.py             # Model pre-download script
├── requirements.txt               # Python dependencies
├── README.md                       # Setup and usage guide
└── INDEX.md                        # This file
```

---

## Files Description

### Core Application Files

#### `rag_chatbot.py` (Main Chatbot)
**Purpose:** Interactive RAG chatbot that retrieves answers from PDF documents

**Key Functions:**
- `load_and_split_documents(pdf_path)` - Loads PDF and splits into 800-char chunks with 120-char overlap
- `build_vector_store(pdf_path, store_dir)` - Creates embeddings and stores in FAISS
- `load_vector_store(store_dir)` - Loads existing vector store from disk
- `create_qa_chain(store_dir)` - Sets up retrieval-QA pipeline
- `main()` - Interactive loop for user questions

**Models Used:**
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: `google/flan-t5-base` (180 token limit)

**Configuration:**
```python
PDF_PATH = "2506.18027v3.pdf"
STORE_DIR = "faiss_store/"
Chunk Size: 800 characters
Chunk Overlap: 120 characters
Search Results: Top 4 most relevant chunks
```

#### `download_models.py` (Model Pre-downloader)
**Purpose:** Downloads and caches required ML models before running chatbot

**Downloads:**
1. Sentence-Transformers embedding model (67MB)
2. Google FLAN-T5-Base LLM (990MB)

**Usage:** Run once to pre-download models for faster chatbot startup

#### `web_chatbot.py` (Browser Chat Interface)
**Purpose:** Serves a local browser UI for the RAG chatbot on `http://127.0.0.1:8000`

**Behavior:**
- starts a simple HTTP server
- loads or builds the FAISS vector store
- exposes a POST `/ask` endpoint for question answering
- returns generated answers from the PDF knowledge base

---

### Configuration & Dependencies

#### `requirements.txt`
**Python Packages:**
- `langchain==0.3.19` - LLM framework
- `langchain-community==0.3.13` - Community tools
- `langchain-huggingface==0.2.0` - HuggingFace integration
- `sentence-transformers==3.2.0` - Embedding model
- `pypdf==4.0.0` - PDF loading
- `transformers==4.46.2` - HuggingFace transformers
- `torch==2.2.2+cpu` - PyTorch CPU
- `faiss-cpu==1.14.3` - Vector search
- `numpy<2` - Numerical computing

#### `.venv/` Virtual Environment
Python 3.11.9 isolated environment with all dependencies

---

### Data & Storage

#### `2506.18027v3.pdf`
**Source Document:** Academic research paper used for knowledge base

#### `faiss_store/`
**Vector Database:**
- Contains pre-built FAISS index from PDF chunks
- Stores embeddings for semantic search
- Binary format (not human-readable)

---

### Documentation

#### `README.md`
- Setup instructions
- Installation steps
- Usage guide
- High-level project description

#### `INDEX.md` (This File)
- Detailed project structure
- File descriptions
- Function references
- Architecture overview

---

## Workflow & Pipeline

```
1. Load PDF (PyPDFLoader)
         ↓
2. Split into Chunks (RecursiveCharacterTextSplitter)
         ↓
3. Generate Embeddings (SentenceTransformer)
         ↓
4. Store in FAISS (Vector Database)
         ↓
5. User Query → Retrieve Top 4 Chunks → FLAN-T5 Generates Answer
```

---

## Data Flow

### Initialization Phase
```
rag_chatbot.py runs → Checks for existing FAISS store
    ├─ If exists: Load existing index (fast)
    └─ If missing: Build new index from PDF
    
    → Initialize embedding model (all-MiniLM-L6-v2)
    → Initialize LLM pipeline (flan-t5-base)
    → Ready for queries
```

### Query Phase
```
User Input → Embed Query → Search FAISS (top 4) → Pass to FLAN-T5 
    → Generate Answer (max 180 tokens) → Display Result
```

---

## Key Components

### Embedding Model
- **Name:** `sentence-transformers/all-MiniLM-L6-v2`
- **Type:** Semantic text embeddings
- **Dimension:** 384-dimensional vectors
- **Purpose:** Convert text chunks to semantic vectors for retrieval

### Language Model  
- **Name:** `google/flan-t5-base`
- **Type:** Sequence-to-sequence transformer
- **Size:** ~250M parameters
- **Max Output:** 180 tokens
- **Purpose:** Generate natural language answers

### Vector Database
- **Name:** FAISS (Facebook AI Similarity Search)
- **Index Type:** Binary flat index
- **Search:** Cosine similarity
- **Purpose:** Fast semantic search over embeddings

---

## Usage Instructions

### 1. First Time Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py  # Pre-download models (10-15 min)
```

### 2. Run Chatbot
```bash
.venv\Scripts\python rag_chatbot.py
```

### 3. Interact
```
Question: What is the main topic of this paper?
Answer: [Retrieved and generated answer]

Question: exit
# Exits the program
```

---

## Dependencies Graph

```
rag_chatbot.py
  ├─ langchain (LLM orchestration)
  ├─ sentence-transformers (embeddings)
  ├─ faiss-cpu (vector search)
  ├─ pypdf (PDF loading)
  └─ transformers (HF models)
     └─ torch (neural compute)

download_models.py
  ├─ transformers (model download)
  └─ sentence-transformers (model download)
```

---

## Performance Notes

- **First Run:** ~2-3 minutes (PDF processing + embeddings)
- **Subsequent Runs:** ~30 seconds (load FAISS index)
- **Query Latency:** 10-15 seconds (FLAN-T5 generation)
- **Memory Usage:** ~2-3 GB (loaded models)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PDF not found | Ensure `2506.18027v3.pdf` is in project root |
| Model download fails | Check internet connection, Hugging Face reachability |
| FAISS deserialization error | Already fixed in code with `allow_dangerous_deserialization=True` |
| Out of memory | Close other applications, use smaller models |

---

## Future Enhancements

- [ ] Support multiple PDF files
- [ ] Web UI (Streamlit/Gradio)
- [ ] Streaming responses
- [ ] Custom model selection
- [ ] Question reformulation
- [ ] Source citation
- [ ] Caching of frequent queries

---

*Last Updated: 2026-06-27*
