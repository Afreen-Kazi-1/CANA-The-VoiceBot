import os
import torch
import faiss
import pickle
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import nltk
nltk.download('punkt_tab')

class RAGPipeline:
    def __init__(self, embedding_model_name="sentence-transformers/LaBSE"):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name)
        self.index = None
        self.content_chunks = []

    def chunk_content_by_sentence(self, text):
        return sent_tokenize(text)

    def load_or_build_index(self, extracted_text, faiss_index_path, content_chunks_path):
        if faiss_index_path.exists() and content_chunks_path.exists():
            try:
                self.index = faiss.read_index(str(faiss_index_path))
                with open(content_chunks_path, 'rb') as f:
                    self.content_chunks = pickle.load(f)
                print("Loaded existing FAISS index and chunks.")
                return
            except Exception as e:
                print(f"Failed to load existing index/chunks: {e}. Rebuilding...")

        print("Chunking text...")
        self.content_chunks = self.chunk_content_by_sentence(extracted_text)

        if not self.content_chunks:
            raise ValueError("No content chunks generated.")

        print(f"Generating embeddings for {len(self.content_chunks)} chunks...")
        chunk_embeddings = []
        for chunk in self.content_chunks:
            chunk = chunk.strip().lower()  # Normalize for multilingual consistency
            inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
            chunk_embeddings.append(embeddings.detach().numpy())

        chunk_embeddings_np = np.vstack(chunk_embeddings)
        dim = chunk_embeddings_np.shape[1]

        print("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(chunk_embeddings_np)

        faiss.write_index(self.index, str(faiss_index_path))
        with open(content_chunks_path, 'wb') as f:
            pickle.dump(self.content_chunks, f)
        print("Saved FAISS index and chunks.")

    def get_relevant_context(self, query, k=5):
        query = query.strip().lower()  # Normalize query too
        query_inputs = self.tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            query_embedding = self.model(**query_inputs).last_hidden_state.mean(dim=1).detach().numpy()

        k = min(k, len(self.content_chunks))
        if k == 0 or self.index is None:
            return []

        distances, indices = self.index.search(query_embedding, k)
        return [self.content_chunks[i] for i in indices[0] if i < len(self.content_chunks)]