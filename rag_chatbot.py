import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, List, cast

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


class LocalTfidfEmbeddings(Embeddings):
    def __init__(self, path: Path | None = None):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.path = path

    def __call__(self, text: str):
        return self.embed_query(text)

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


def create_qa_chain(store_dir: Path):
    vectorstore = load_vector_store(store_dir)
    return LocalRAG(vectorstore)


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found at {PDF_PATH}")
        sys.exit(1)

    if STORE_DIR.exists():
        print("Using existing FAISS vector store from faiss_store.")
    else:
        print("Building vector store from PDF...")
        build_vector_store(PDF_PATH, STORE_DIR)

    qa = create_qa_chain(STORE_DIR)
    print("RAG chatbot ready. Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        print("\nAnswer:\n" + qa.run(question) + "\n")


if __name__ == "__main__":
    main()
