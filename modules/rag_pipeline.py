import os
import faiss
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import re
import requests
import json

# Load env and Groq
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_AI_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
llama_model = "llama-3.3-70b-versatile"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Tokenizers and Models (needed only for query embedding)
labse_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

mpnet_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
mpnet_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def detect_script(text):
    devanagari_count = sum('\u0900' <= c <= '\u097F' for c in text)
    latin_count = sum('a' <= c.lower() <= 'z' for c in text)
    if devanagari_count > 0:
        return "Hindi"
    elif latin_count > 0:
        return "Latin"
    else:
        return "Unknown"

def generate_rag_response_from_text(query, tokenizer, model, index, content_chunks): #
    query_inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        query_embedding = model(**query_inputs).last_hidden_state.mean(dim=1).detach().numpy()

    k = min(5, len(content_chunks))
    if k == 0:
        return "No content available to search."

    distances, indices = index.search(query_embedding, k)
    relevant_contexts = [content_chunks[i] for i in indices[0] if i < len(content_chunks)]
    combined_context = " ".join(relevant_contexts)

    if not combined_context.strip():
        return "Could not find relevant context for your query."

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "model": llama_model,
        "messages": [{
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that answers questions based on the provided context "
                            "from PDF documents. If the answer is not found in the context, say "
                            "'So sorry, I could not find an answer to that.' Be concise with replies. "
                            "Give suggestions where appropriate. Use conversational nuances and avoid overly technical jargon unless necessary."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context from PDF documents:\n{combined_context}\n\nQuestion: {query}"
                    }
            ]
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        command = result["choices"][0]["message"]["content"].strip()
        return command
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Error interpreting command."
    
def rag_model(input, language): # for middleman
    working_dir = Path("./rag_cache")
    labse_chunks_path = working_dir / "labse_content_chunks.pkl"
    labse_index_path = working_dir / "labse_faiss.index"

    mpnet_chunks_path = working_dir / "mpnet_content_chunks.pkl"
    mpnet_index_path = working_dir / "mpnet_faiss.index"
    if language == "Hindi":
        return generate_rag_response_from_text(
            input, labse_tokenizer, labse_model, labse_index_path, labse_chunks_path
        )
    elif language == "Latin":
        return generate_rag_response_from_text(
            input, mpnet_tokenizer, mpnet_model, mpnet_index_path, mpnet_chunks_path
        )
    else:
        return "Language not supported or could not be determined."

def run_rag_pipeline(pdf_dir, test_csv_path, output_csv_path): #for inference
    working_dir = Path("./rag_cache")
    # Precomputed chunk and embedding file paths
    labse_chunks_path = working_dir / "labse_content_chunks.pkl"
    labse_index_path = working_dir / "labse_faiss.index"

    mpnet_chunks_path = working_dir / "mpnet_content_chunks.pkl"
    mpnet_index_path = working_dir / "mpnet_faiss.index"

    # Load precomputed chunks and FAISS indices
    if not all(p.exists() for p in [labse_chunks_path, labse_index_path, mpnet_chunks_path, mpnet_index_path]):
        raise FileNotFoundError("Missing required precomputed files in ./rag_cache")

    with open(labse_chunks_path, 'rb') as f:
        labse_chunks = pickle.load(f)
    with open(mpnet_chunks_path, 'rb') as f:
        mpnet_chunks = pickle.load(f)

    labse_index = faiss.read_index(str(labse_index_path))
    mpnet_index = faiss.read_index(str(mpnet_index_path))

    # Load input questions
    df = pd.read_csv(test_csv_path)
    normalized_cols = [col.strip().lower() for col in df.columns]
    if "questions" not in normalized_cols:
        raise ValueError("Test CSV must contain a 'Questions' column (case-insensitive).")
    actual_col = df.columns[normalized_cols.index("questions")]

    # Answer generation
    responses = []
    for question in df[actual_col]:
        lang = detect_script(question)
        if lang == "Hindi":
            answer = generate_rag_response_from_text(
                question, labse_tokenizer, labse_model, labse_index, labse_chunks
            )
        elif lang == "Latin":
            answer = generate_rag_response_from_text(
                question, mpnet_tokenizer, mpnet_model, mpnet_index, mpnet_chunks
            )
        else:
            answer = "Language not supported or could not be determined."
        responses.append(answer)

    df["Responses"] = responses
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Responses saved to: {output_csv_path}")
