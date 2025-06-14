from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
GROQ_API_KEY = os.getenv("GROQ_AI_API_KEY")
llama_model = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

def generate_response(query, contexts, model_name=llama_model, temperature=0.2):
    if not contexts:
        return "No content available to search."

    combined_context = " ".join(contexts)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are a helpful assistant that answers questions based on the provided context from PDF documents.
                        If the answer is not found in the context, say 'I could not find an answer to that in the provided documents.'
                        Be concise and directly answer the question based on the information given.
                    """
                },
                {
                    "role": "user",
                    "content": f"Context from PDF documents:\n{combined_context}\n\nQuestion: {query}"
                }
            ],
            model=model_name,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"