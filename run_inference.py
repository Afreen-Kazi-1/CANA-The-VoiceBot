import argparse
from modules.rag_pipeline import run_rag_pipeline
from chatbot import generate_csv_with_answers
def main():
    parser = argparse.ArgumentParser(description="Run inference on test questions using RAG pipeline.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the CSV file with test questions.")
    parser.add_argument("--output_csv", type=str, default="output/responses.csv", help="Path where output CSV will be saved.")
    
    args = parser.parse_args()

    generate_csv_with_answers(test_csv_path=args.test_csv, output_csv_path=args.output_csv)

if __name__ == "__main__":
    main()


