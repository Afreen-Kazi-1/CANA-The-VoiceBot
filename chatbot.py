import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# nltk.download('punkt_tab') # This was likely an error, 'punkt' is usually sufficient.
# The line above was corrected from nltk.download('punkt_tab') to just ensure 'punkt' is available if needed, or use "llama3-groq-8b-8192-tool-use-preview".
# However, sent_tokenize itself should handle this. If issues persist, uncommenting nltk.download('punkt') might be necessary.

from nltk.tokenize import sent_tokenize
import os
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import fitz 
from groq import Groq
from pathlib import Path
import glob
import pickle
import pandas as pd
from dotenv import load_dotenv

# --- Global Configuration & Model Initialization ---
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
client = Groq(api_key=GROQ_API_KEY)
llama_model = "llama-3.3-70b-versatile" 

# Initialize tokenizer and embedding model globally
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    embedding_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    EMBEDDING_DIM = embedding_model.config.hidden_size
except Exception as e:
    print(f"Error initializing HuggingFace models: {e}")
    print("Please ensure you have an internet connection and the model name is correct.")
    tokenizer = None
    embedding_model = None
    EMBEDDING_DIM = 768 # Default, but will cause issues if model not loaded

# Global variables for RAG artifacts
faiss_index = None
content_chunks = None
rag_artifacts_loaded = False

# --- PDF Processing and Chunking (kept for potential future use, not called in primary flow) ---
def extract_text_from_pdf(pdf_file_path):
    """Extract text from a single PDF file."""
    doc = fitz.open(pdf_file_path)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        all_text += f"--- Page {page_num + 1} of {os.path.basename(pdf_file_path)} ---\n{page_text}\n"
    doc.close()
    return all_text

def extract_text_from_pdf_directory(pdf_directory):
    """Extract text from all PDF files in a given directory."""
    all_docs_text = ""
    pdf_files = glob.glob(os.path.join(pdf_directory, '*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in directory: {pdf_directory}")
        return ""
    
    print(f"Found {len(pdf_files)} PDF files to process.")
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        try:
            all_docs_text += extract_text_from_pdf(pdf_file) + "\n\n"
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    return all_docs_text

def chunk_content_by_sentence(text):
    """Chunk content into sentences."""
    return sent_tokenize(text)

# --- RAG Artifact Loading ---
def load_rag_artifacts():
    global faiss_index, content_chunks, rag_artifacts_loaded
    current_script_dir = Path(__file__).parent
    faiss_index_path = current_script_dir / "rag_faiss.index"
    content_chunks_path = current_script_dir / "rag_content_chunks.pkl"

    print("Attempting to load RAG artifacts...")
    try:
        if faiss_index_path.exists():
            faiss_index = faiss.read_index(str(faiss_index_path))
            print(f"Successfully loaded FAISS index from {faiss_index_path} with {faiss_index.ntotal} vectors.")
        else:
            print(f"Error: FAISS index file not found at {faiss_index_path}")
            faiss_index = None

        if content_chunks_path.exists():
            with open(content_chunks_path, 'rb') as f:
                content_chunks = pickle.load(f)
            print(f"Successfully loaded content chunks from {content_chunks_path} ({len(content_chunks)} chunks).")
        else:
            print(f"Error: Content chunks file not found at {content_chunks_path}")
            content_chunks = None
            
        if faiss_index is not None and content_chunks is not None:
            rag_artifacts_loaded = True
            print("RAG artifacts loaded successfully.")
        else:
            rag_artifacts_loaded = False
            print("Failed to load one or more RAG artifacts. RAG functions will not operate correctly.")

    except Exception as e:
        print(f"An error occurred while loading RAG artifacts: {e}")
        faiss_index = None
        content_chunks = None
        rag_artifacts_loaded = False

# Load artifacts when the module is imported/run
load_rag_artifacts()

# --- Core RAG Functions ---
def get_bot_response(query: str) -> str:
    """
    Generates a RAG response for a given query using pre-loaded artifacts.
    Uses global client, llama_model, tokenizer, embedding_model, faiss_index, content_chunks.
    """
    if not rag_artifacts_loaded or tokenizer is None or embedding_model is None or faiss_index is None or content_chunks is None:
        return "Error: RAG components are not properly loaded. Cannot generate response."
    if not query or not query.strip():
        return "Error: Query cannot be empty."

    try:
        query_inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            query_embedding = embedding_model(**query_inputs).last_hidden_state.mean(dim=1).detach().numpy()
        
        k = min(5, len(content_chunks)) 
        if k == 0:
            return "No content available in loaded chunks to search."
            
        distances, indices = faiss_index.search(query_embedding, k)
        
        relevant_contexts = [content_chunks[i] for i in indices[0] if i < len(content_chunks)]
        combined_context = " ".join(relevant_contexts)

        if not combined_context.strip():
            return "Could not find relevant context for your query in the loaded documents."

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You give a very natural human like responses. if the user is talking causally, you talk casually as well. if the user is talking in hinglish, you talk in hinglish too. you are not a bot, you are like a human, talk like a normal human would talk. You are a knowledgeable expert who answers questions based only on the provided content from PDF documents.
Speak naturally and directly, as if you're the author of the documents.
Do not mention "the documents", "context", or similar references.
If the answer is not available, simply ask what the user wants again.
understand the user's sentiments and respond accordingly.
add punctuation to your responses. add exclamation marks, question marks, commas, and full stops where appropriate. 
be more emphatetic when the user's response is negative or sad.
Keep your answers concise, clear, and helpful."""
                },
                {
                    "role": "user",
                    "content": f"Context from PDF documents:\n{combined_context}\n\nQuestion: {query}"
                }
            ],
            model=llama_model,
        )
        answer = chat_completion.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Error during RAG response generation for query '{query}': {e}")
        return "Error generating response from LLM."

def generate_csv_with_answers(input_csv_path: str, output_csv_path: str):
    """
    Reads questions from an input CSV, generates answers using get_bot_response,
    and writes questions and answers to an output CSV.
    Uses global RAG components.
    """
    if not rag_artifacts_loaded:
        print("Error: RAG components are not loaded. Cannot process CSV.")
        return

    try:
        df = pd.read_csv(input_csv_path)
        if 'questions' not in df.columns:
            print(f"Error: 'questions' column not found in {input_csv_path}")
            return
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}.")
        return
    except Exception as e:
        print(f"Error reading CSV {input_csv_path}: {e}")
        return

    answers = []
    total_questions = len(df)
    print(f"Processing {total_questions} questions from {input_csv_path}...")
    
    for i, row in df.iterrows():
        question = str(row['questions']) 
        if not question.strip():
            print(f"Skipping empty question at row {i+1}.")
            answers.append("Skipped empty question.")
            continue

        print(f"Processing question {i+1}/{total_questions}: \"{question[:70]}...\"")
        
        answer = get_bot_response(question)
        answers.append(answer)
        print(f"  -> Answer generated: \"{str(answer)[:70]}...\"")

    df['answers'] = answers

    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Successfully processed questions and saved results to {output_csv_path}")
    except Exception as e:
        print(f"Error writing CSV to {output_csv_path}: {e}")

# Example usage (optional, can be commented out or removed if this is purely a library)
# if __name__ == "__main__":
#     print("Chatbot module loaded. RAG artifacts status:", rag_artifacts_loaded)
#     if rag_artifacts_loaded:
#         # Test get_bot_response
#         print("\n--- Testing single response ---")
#         sample_query = "What is LenDenClub?"
#         print(f"Query: {sample_query}")
#         bot_answer = get_bot_response(sample_query)
#         print(f"Bot Answer: {bot_answer}")

#         # Test generate_csv_with_answers
#         print("\n--- Testing CSV processing ---")
#         # Create a dummy input CSV for testing if it doesn't exist
#         test_input_csv = "test_questions.csv"
#         if not os.path.exists(test_input_csv):
#             try:
#                 dummy_df = pd.DataFrame({'questions': ["What are P2P investments?", "Tell me about RBI guidelines."]})
#                 dummy_df.to_csv(test_input_csv, index=False)
#                 print(f"Created dummy input file: {test_input_csv}")
#             except Exception as e:
#                 print(f"Could not create dummy input file: {e}")

#         if os.path.exists(test_input_csv):
#             generate_csv_with_answers(test_input_csv, "test_questions_with_answers.csv")
#         else:
#             print(f"Skipping CSV processing test as '{test_input_csv}' was not found/created.")
#     else:
#         print("Cannot run examples because RAG artifacts were not loaded.")
#         print("Please ensure 'rag_faiss.index' and 'rag_content_chunks.pkl' are in the same directory as the script.")

# (The old main() function and its direct calls to process_csv_and_generate_answers have been removed)
# (The old process_csv_and_generate_answers and generate_rag_response_from_text functions have been renamed and adapted)