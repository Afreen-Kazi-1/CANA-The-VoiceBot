import os
from pathlib import Path
from modules.nlp_pipeline import RAGPipeline
from modules.response_gen import generate_response
from modules.utils import extract_text_from_pdf_directory  # You need this utility to extract text from PDF files


def main():
    pdf_directory = input("Enter the path to the directory containing PDF files: ")
    if not os.path.isdir(pdf_directory):
        print("Invalid directory path.")
        return

    # Define paths to store the FAISS index and chunk data
    current_script_dir = Path(__file__).parent
    faiss_index_path = current_script_dir / "rag_faiss.index"
    content_chunks_path = current_script_dir / "rag_content_chunks.pkl"

    print("Extracting text from PDF documents...")
    all_text = extract_text_from_pdf_directory(pdf_directory)

    if not all_text.strip():
        print("No text found in PDF documents. Exiting.")
        return

    # Initialize the pipeline
    rag_pipeline = RAGPipeline()
    rag_pipeline.load_or_build_index(all_text, faiss_index_path, content_chunks_path)

    print("\nPDF RAG Chatbot is ready. Ask your questions (type 'exit' to quit).")
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break

        if not query.strip():
            print("Bot: Please enter a question.")
            continue

        contexts = rag_pipeline.get_relevant_context(query)
        response = generate_response(query, contexts)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
