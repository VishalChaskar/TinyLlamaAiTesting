import os
import faiss
import torch
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and fast

# Global cache
index = None
documents = []

def load_pdf_chunks(pdf_path, chunk_size=300):
    reader = PdfReader(pdf_path)
    all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]
    return chunks

def build_faiss_index(chunks):
    global index, documents
    embeddings = embedder.encode(chunks, convert_to_tensor=True).cpu().numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    documents = chunks

def prepare_rag_context(query, pdf_path="local_data/example.pdf", top_k=3):
    global index, documents

    if index is None:
        chunks = load_pdf_chunks(pdf_path)
        build_faiss_index(chunks)

    query_vec = embedder.encode([query], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(query_vec, top_k)

    top_chunks = [documents[i] for i in I[0]]
    return "\n".join(top_chunks)
