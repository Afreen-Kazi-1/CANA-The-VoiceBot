import tkinter as tk
from ui import TranscriptionApp
from chatbot import get_bot_response
from middleman import middleman

# Global variables to store user input and language
user_input = ""
language = "en-US"
data = ""
context = []
system_out = ""

def update_user_data(text, lang):
    """Update global variables with user input and language"""
    global user_input, language, data, system_out
    user_input = text
    language = lang
    
    # Get RAG data using the user input
    if user_input:
        data = get_bot_response(user_input)
        # Call middleman function with user_input, context, and data
        system_out = middleman(user_input, context, data)
        # Print the system output to console
        print(f"System Output: {system_out}")

def main():
    """Main function to start the transcription app"""
    root = tk.Tk()
    app = TranscriptionApp(root)
    
    # Store reference to update function for potential future use
    app.update_user_data = update_user_data
    
    root.mainloop()

if __name__ == "__main__":
    main()