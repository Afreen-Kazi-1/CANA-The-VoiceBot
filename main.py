import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('punkt_tab')  # Ensure punkt tokenizer is downloaded
    
# nltk.download('punkt_tab') # This might be an error, 'punkt' is usually sufficient.
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
from dotenv import load_dotenv
import pickle # Added for saving/loading chunks

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_AI_API_KEY")  # Keep your API key secure
client = Groq(api_key=GROQ_API_KEY)

llama_model = "llama-3.3-70b-versatile" # Corrected model name based on previous context if needed, or use "llama3-groq-8b-8192-tool-use-preview"

# base_dir = 'extracted_images/' # Removed: No longer storing extracted images
# pdf_dir = os.path.join(base_dir, 'pdfs') # Removed
# web_dir = os.path.join(base_dir, 'webscraping') # Removed

# os.makedirs(pdf_dir, exist_ok=True) # Removed
# os.makedirs(web_dir, exist_ok=True) # Removed

def extract_text_from_pdf(pdf_file_path):
    """Extract text from a single PDF file."""
    doc = fitz.open(pdf_file_path)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        all_text += f"--- Page {page_num + 1} of {os.path.basename(pdf_file_path)} ---\\n{page_text}\\n"
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
            all_docs_text += extract_text_from_pdf(pdf_file) + "\\n\\n"
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    return all_docs_text

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def chunk_content_by_sentence(text):
    """Chunk content into sentences."""
    return sent_tokenize(text)

def generate_rag_response_from_text(query, client, llama_model_name, tokenizer, embedding_model, index, content_chunks):
    query_inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        query_embedding = embedding_model(**query_inputs).last_hidden_state.mean(dim=1).detach().numpy()
    
    k = min(5, len(content_chunks)) # Number of relevant contexts to retrieve, ensure k is not greater than available chunks
    if k == 0:
        return "No content available to search."
        
    distances, indices = index.search(query_embedding, k)
    
    relevant_contexts = [content_chunks[i] for i in indices[0] if i < len(content_chunks)]
    combined_context = " ".join(relevant_contexts)

    if not combined_context.strip():
        return "Could not find relevant context for your query."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that answers questions based on the provided context from PDF documents.
                    If the answer is not found in the context, say 'I could not find an answer to that in the provided documents.'
                    Be concise and directly answer the question based on the information given."""
                },
                {
                    "role": "user",
                    "content": f"Context from PDF documents:\\n{combined_context}\\n\\nQuestion: {query}"
                }
            ],
            model=llama_model_name,
            temperature=0.2, # Adjusted for more factual responses
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    pdf_directory = input("Enter the path to the directory containing PDF files: ")
    if not os.path.isdir(pdf_directory):
        print("Invalid directory path.")
        return

    # Define paths for saved index and chunks
    # Ensure these paths are appropriate for your system
    current_script_dir = Path(__file__).parent
    faiss_index_path = current_script_dir / "rag_faiss.index"
    content_chunks_path = current_script_dir / "rag_content_chunks.pkl"
    
    # Attempt to load existing index and chunks
    if faiss_index_path.exists() and content_chunks_path.exists():
        print("Loading existing FAISS index and content chunks...")
        try:
            index = faiss.read_index(str(faiss_index_path))
            with open(content_chunks_path, 'rb') as f:
                content_chunks = pickle.load(f)
            print("Successfully loaded index and chunks.")
        except Exception as e:
            print(f"Error loading existing files: {e}. Re-processing PDFs.")
            index = None
            content_chunks = None
    else:
        print("No existing index/chunks found or one is missing. Processing PDFs.")
        index = None
        content_chunks = None

    if index is None or content_chunks is None:
        print("Extracting text from PDF documents...")
        all_documents_text = extract_text_from_pdf_directory(pdf_directory)

        if not all_documents_text.strip():
            print("No text could be extracted from the PDF documents. Exiting.")
            return

        print("Text extraction complete. Chunking content...")
        content_chunks = chunk_content_by_sentence(all_documents_text)
        
        if not content_chunks:
            print("No content chunks to process. Exiting.")
            return

        print(f"Created {len(content_chunks)} chunks. Generating embeddings...")
        
        # Generate embeddings for content chunks
        chunk_embeddings = []
        for chunk in content_chunks:
            inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512) # Added max_length
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1)
            chunk_embeddings.append(embeddings.detach().numpy())
        
        if not chunk_embeddings:
            print("Failed to generate embeddings. Exiting.")
            return
            
        chunk_embeddings_np = np.vstack(chunk_embeddings)

        print("Embeddings generated. Building FAISS index...")
        dimension = chunk_embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings_np)
        print("FAISS index built successfully.")

        # Save the newly created index and chunks
        print("Saving FAISS index and content chunks...")
        try:
            faiss.write_index(index, str(faiss_index_path))
            with open(content_chunks_path, 'wb') as f:
                pickle.dump(content_chunks, f)
            print("Successfully saved index and chunks.")
        except Exception as e:
            print(f"Error saving index/chunks: {e}")

    print("\nPDF RAG Chatbot is ready. Ask your questions (type 'exit' to quit).")
    while True:
        user_query = input("\\nYou: ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            print("Bot: Please enter a question.")
            continue
        
        response = generate_rag_response_from_text(user_query, client, llama_model, tokenizer, model, index, content_chunks)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()