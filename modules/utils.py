# === modules/pdf_utils.py ===
import os
import glob
import fitz  # PyMuPDF

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
