import json
import pickle
import shutil
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

PDF_PATH = Path(__file__).resolve().parent / "2506.18027v3.pdf"
STORE_DIR = Path(__file__).resolve().parent / "faiss_store"
EMBEDDING_PATH = STORE_DIR / "tfidf_vectorizer.pkl"
HOST = "127.0.0.1"
PORT = 8000


class LocalTfidfEmbeddings(Embeddings):
    def __init__(self, path: Path | None = None):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.path = path

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.save()
        return self

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not hasattr(self.vectorizer, "vocabulary_"):
            self.vectorizer.fit(texts)
        matrix = cast(Any, self.vectorizer.transform(texts))
        return np.asarray(matrix.toarray(), dtype=float).tolist()

    def embed_query(self, text: str) -> List[float]:
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise ValueError("Embedding model is not fitted yet.")
        matrix = cast(Any, self.vectorizer.transform([text]))
        return np.asarray(matrix.toarray(), dtype=float).tolist()[0]

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as handle:
            pickle.dump(self.vectorizer, handle)

    def load(self):
        if self.path is not None and self.path.exists():
            with self.path.open("rb") as handle:
                self.vectorizer = pickle.load(handle)
        return self


INDEX_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>PDF RAG Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; max-width: 860px; }
    h1 { margin-bottom: 0.25rem; }
    label, button { display: block; margin-top: 1rem; }
    textarea { width: 100%; min-height: 120px; padding: 0.75rem; font-size: 1rem; }
    #answer { margin-top: 1rem; padding: 1rem; background: #f5f5f5; border: 1px solid #ddd; white-space: pre-wrap; min-height: 120px; }
    #status { margin-top: 0.75rem; font-weight: bold; }
  </style>
</head>
<body>
  <h1>PDF RAG Chatbot</h1>
  <p>Ask questions about the PDF <strong>2506.18027v3.pdf</strong>.</p>
  <form id=\"chat-form\">
    <label for=\"question\">Question</label>
    <textarea id=\"question\" placeholder=\"Enter your question here...\"></textarea>
    <button type=\"submit\">Ask</button>
  </form>
  <div id=\"status\"></div>
  <div id=\"answer\"></div>
  <script>
    const form = document.getElementById('chat-form');
    const status = document.getElementById('status');
    const answer = document.getElementById('answer');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const question = document.getElementById('question').value.trim();
      if (!question) {
        status.textContent = 'Please type a question.';
        return;
      }
      status.textContent = 'Thinking...';
      answer.textContent = '';

      try {
        const response = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question }),
        });
        const body = await response.json();
        if (response.ok) {
          status.textContent = 'Answer received.';
          answer.textContent = body.answer;
        } else {
          status.textContent = 'Error: ' + (body.error || 'Unknown error');
        }
      } catch (error) {
        status.textContent = 'Request failed: ' + error.message;
      }
    });
  </script>
</body>
</html>"""


def load_and_split_documents(pdf_path: Path) -> List[Document]:
    reader = PdfReader(str(pdf_path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("No text could be loaded from the PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": str(pdf_path)}) for chunk in chunks]


def build_vector_store(pdf_path: Path, store_dir: Path):
    docs = load_and_split_documents(pdf_path)
    if not docs:
        raise ValueError("No text could be loaded from the PDF.")

    if store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    embeddings = LocalTfidfEmbeddings(EMBEDDING_PATH)
    embeddings.fit([doc.page_content for doc in docs])
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local(str(store_dir))
    return vectorstore


def load_vector_store(store_dir: Path):
    embeddings = LocalTfidfEmbeddings(EMBEDDING_PATH).load()
    return FAISS.load_local(str(store_dir), embeddings, allow_dangerous_deserialization=True)


class LocalRAG:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def run(self, question: str) -> str:
        docs = self.vectorstore.similarity_search(question, k=4)
        if not docs:
            return "No relevant content was found in the PDF."

        context = "\n\n".join(doc.page_content for doc in docs)
        snippet = context[:1200].strip()
        if len(context) > 1200:
            snippet += "..."
        return f"Based on the most relevant passages from the PDF:\n\n{snippet}"


class ChatHandler(BaseHTTPRequestHandler):
    def _send_html(self, html: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _send_json(self, data: Dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path != "/ask":
            self.send_error(404, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(body)
            question = payload.get("question", "").strip()
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON payload"}, status_code=400)
            return

        if not question:
            self._send_json({"error": "Question is required"}, status_code=400)
            return

        try:
            answer = self.server.qa.run(question)
            self._send_json({"answer": answer})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status_code=500)

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_qa_chain(store_dir: Path):
    vectorstore = load_vector_store(store_dir)
    return LocalRAG(vectorstore)


def main() -> None:
    if not PDF_PATH.exists():
        print(f"PDF not found at {PDF_PATH}")
        sys.exit(1)

    if STORE_DIR.exists():
        print("Using existing FAISS vector store from faiss_store.")
    else:
        print("Building vector store from PDF...")
        build_vector_store(PDF_PATH, STORE_DIR)

    print("Loading QA chain...")
    qa = create_qa_chain(STORE_DIR)

    server = HTTPServer((HOST, PORT), ChatHandler)
    server.qa = qa

    print(f"Open your browser and go to http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
