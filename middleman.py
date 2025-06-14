import os
import requests
import json
import re
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

a = "SENTENCE: RBI Stands for Reserve Bank of India; USER_INPUT: Are you stupid? Whais RBI?; SENTIMENT: Negative; INTENT: Anger; The user is asking the USER_INPUT with the given SENTIMENT and INTENT. As a responder, Rephrase the SENTENCE to be a perfect response to the user's question. The user should be satisfied, and you should always calm the user down. Keep it maximum 2 lines"

def interpret_command_with_api(user_input):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [{
            "role": "user",
            "content": a
        }]
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        command = result["choices"][0]["message"]["content"].strip()
        command = re.sub(r"```(bash|shell)?", "", command).strip()
        return command
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Error interpreting command."
    
b= interpret_command_with_api(a)
print(b)

def middleman(user_input, context, data, sa):
    a = f"SENTENCE: {data}; USER_INPUT: {user_input}; SENTIMENT: {sa['sentiment']}; INTENT: {sa['intent']}; CONFIDENCE_SCORE: {sa['confidence score']}. The user is asking the USER_INPUT with the given SENTIMENT and INTENT. If the confidence score is below 0.5, ask the user again, what he wants. As a responder, Rephrase the SENTENCE to be a perfect response to the user's question. The user should be satisfied, and you should always calm the user down. Keep it maximum 2 lines"

    command = interpret_command_with_api(a)
    if command.startswith("Error"):
        return command
    return command
