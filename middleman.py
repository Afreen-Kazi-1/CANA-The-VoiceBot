import os
import requests
import json
import re
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def interpret_command_with_api(user_input):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [{
            "role": "user",
            "content": user_input
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


def middleman(user_input, context, data):
    sentence = f"USER_INPUT: {user_input}; SYSTEM_OUTPUT: {data}; . The user is asking the USER_INPUT with some sentiments and intent. As a responder, Rephrase the SYSTEM_OUTPUT to be a perfect response to the user's question. The user should be satisfied, and you should always calm the user down. Keep it maximum 2 lines"

    command = interpret_command_with_api(sentence)
    if command.startswith("Error"):
        return command
    return command
