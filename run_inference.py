# === run_inference.py ===
import pandas as pd
import argparse
from pathlib import Path
from modules.nlp_pipeline import RAGPipeline
from modules.response_gen import generate_response
from modules.utils import extract_text_from_pdf_directory


def main():
    parser = argparse.ArgumentParser(description="Run inference on test questions for RAG-based voicebot")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Path to directory containing PDF files")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to CSV file with test questions")
    parser.add_argument("--output_csv", type=str, default="output/responses.csv", help="Path to save the output CSV")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    test_csv_path = Path(args.test_csv)
    output_csv_path = Path(args.output_csv)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    print("Extracting content from PDFs...")
    extracted_text = extract_text_from_pdf_directory(str(pdf_dir))
    if not extracted_text.strip():
        raise ValueError("No content extracted from PDFs")

    print("Initializing RAG pipeline...")
    rag = RAGPipeline()

    base_path = Path(__file__).parent
    faiss_index_path = base_path / "rag_faiss.index"
    content_chunks_path = base_path / "rag_content_chunks.pkl"

    rag.load_or_build_index(
        extracted_text,
        faiss_index_path=faiss_index_path,
        content_chunks_path=content_chunks_path,
    )

    print("Reading test questions...")
    df = pd.read_csv(test_csv_path)
    normalized_cols = [col.strip().lower() for col in df.columns]
    if "questions" not in normalized_cols:
        raise ValueError("Test CSV must contain a 'Questions' column (case-insensitive)")

    actual_col = df.columns[normalized_cols.index("questions")]

    responses = []
    print("Generating responses...")
    for q in df[actual_col]:
        context = rag.get_relevant_context(q)
        answer = generate_response(q, context)
        responses.append(answer)

    df["Responses"] = responses
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Responses written to: {output_csv_path}")


if __name__ == "__main__":
    main()
