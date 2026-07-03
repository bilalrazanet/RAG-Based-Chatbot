"""
Pre-download models for RAG chatbot
This script downloads the embedding and LLM models in advance
"""
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("Starting model downloads...\n")

# Download embedding model
print("1️⃣ Downloading embedding model: sentence-transformers/all-MiniLM-L6-v2")
embeddings_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Embedding model downloaded!\n")

# Download LLM model
print("2️⃣ Downloading LLM model: google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.float32)
print("✅ LLM model downloaded!\n")

print("🎉 All models are ready! You can now run: python rag_chatbot.py")
