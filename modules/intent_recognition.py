import json
import os
from transformers import pipeline
import pandas as pd
import numpy as np

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load HuggingFace pipelines
zero_shot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=0)
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/xlm-roberta-base-sentiment-multilingual", device=0)

INTENTS = [
    "greeting",
    "acknowledgment",
    "connection_issue",
    "inquiry",
    "callback_inquiry",
    "guidance_request",
    "loan_limit_inquiry",
    "credit_score_inquiry",
    "platform_inquiry",
    "account_inquiry",
    "clarification",
    "representation"
]

# Placeholder RAG function with expanded Hindi/Hinglish FAQs
def rag_generate_response(query):
    faq_dict = {
        "what is the crif score, the cibil score?": "CRIF and CIBIL are credit bureaus. A CRIF score of 500-600 indicates a good profile.",
        "kya interest rate hai?": "Interest rate aapke credit profile ke hisaab se vary karta hai, typically 6-20% APR.",
        "what is the platform again?": "The platform is Instamoney, a LendingClub product.",
        "who is asking for the money, from which platform they're asking?": "Borrowers Instamoney ke through apply karte hain, jo LendingClub ka loan platform hai.",
        "and so the limit of lending 1 person is only 4 1000.": "Ek person ke liye lending limit 4,000 hai, lekin lump-sum plans mein 5,000 ho sakta hai.",
        "no, sir, i haven’t created the account yet.": "Aap lendingclub.com ya Instamoney app par account create kar sakte hain.",
        "so should i arrange a callback for him?": "Haan, callback arrange kiya ja sakta hai. Please details dijiye.",
        "loan ka process kya hai?": "LendingClub par apply karein, credit profile check hoga, aur loan approve ho sakta hai.",
        "cibil score kaise check karu?": "Aap Instamoney app ya LendingClub website par apna CIBIL score check kar sakte hain.",
        "kya loan jaldi mil sakta hai?": "Haan, agar aapka credit profile accha hai, toh loan jaldi approve ho sakta hai."
    }
    return faq_dict.get(query.lower(), f"Generated response for: {query}")

# Load transcript JSON file
def load_transcript(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict) and "segments" in data:
            data = data["segments"]
        if not isinstance(data, list):
            raise ValueError("Transcript JSON must be a list or contain a 'segments' list")
        return data
    except Exception as e:
        print(f"Error loading transcript: {e}")
        return []

# Extract all user queries
def get_user_queries(transcript):
    try:
        return [entry["text"].strip() for entry in transcript if isinstance(entry, dict) and entry.get("speaker_id") == "speaker_1" and entry["text"].strip() != "..."]
    except Exception as e:
        print(f"Error extracting queries: {e}")
        return []

# Multi-intent recognition with highest confidences
def detect_intents(query, confidence_margin=0.1, ambiguity_threshold=0.7):
    try:
        result = zero_shot_classifier(query, candidate_labels=INTENTS, multi_label=True)
        intents = []
        confidences = []
        max_confidence = max(result["scores"])
        
        for label, score in zip(result["labels"], result["scores"]):
            if score >= (max_confidence - confidence_margin):
                intents.append(label)
                confidences.append(round(score, 4))
        
        if not intents or (max_confidence < ambiguity_threshold and sum(confidences) < 1.0):
            return ["ambiguous"], [max(result["scores"])]
        
        return intents, confidences
    except Exception as e:
        print(f"Error in intent detection for '{query}': {e}")
        return ["ambiguous"], [0.0]

# Sentiment analysis and tone adjustment
def analyze_sentiment_and_adjust_tone(query, response):
    try:
        sentiment_result = sentiment_analyzer(query)[0]
        sentiment = sentiment_result["label"].lower()
        score = sentiment_result["score"]
        sentiment_label = sentiment.upper()
        if sentiment_label == "POSITIVE":
            return f"Great to hear! {response}", sentiment_label, score
        elif sentiment_label == "NEGATIVE":
            return f"We are sorry for any trouble caused. {response}", sentiment_label, score
        return response, sentiment_label, score
    except Exception as e:
        print(f"Error in sentiment analysis for '{query}': {e}")
        return response, "NEUTRAL", 0.0

# Full NLP pipeline for all queries
def nlp_pipeline(json_file, output_file):
    transcript = load_transcript(json_file)
    queries = get_user_queries(transcript)
    if not queries:
        results = [{"query": "", "intents": ["none"], "confidences": [0.0], "sentiment": "none", "sentiment_score": 0.0, "response": "No user input detected."}]
    else:
        results = []
        for query in queries:
            intents, confidences = detect_intents(query)
            if "ambiguous" in intents:
                result = {
                    "query": query,
                    "intents": ["ambiguous"],
                    "confidences": confidences,
                    "sentiment": "none",
                    "sentiment_score": 0.0,
                    "response": "Maaf kijiye, samajh nahi aaya. Kya aap spasht kar sakte hain?"
                }
            else:
                rag_response = rag_generate_response(query)
                final_response, sentiment, sentiment_score = analyze_sentiment_and_adjust_tone(query, rag_response)
                result = {
                    "query": query,
                    "intents": intents,
                    "confidences": confidences,
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "response": final_response
                }
            results.append(result)
    
    # Save results to output JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Output saved to {output_file}")
    except Exception as e:
        print(f"Error saving output to {output_file}: {e}")
    
    return results

# Main execution
if __name__ == "__main__":
    json_file = "/content/transcript.json"
    output_file = "/content/output.json"
    try:
        results = nlp_pipeline(json_file, output_file)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error running pipeline: {e}")