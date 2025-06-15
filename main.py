import tkinter as tk
from modules.ui import TranscriptionApp
from modules.rag_pipeline import get_bot_response
from modules.middleman import middleman
from modules.tts import save_audio_from_text
import os
import pygame
import tempfile

# Global variables to store user input and language
user_input = ""
language = "en-US"
data = ""
context = []
system_out = ""

def update_user_data(text, lang):
    """Update global variables with user input and language"""
    global user_input, language, data, system_out, context
    user_input = text
    language = lang
    
    # Get RAG data using the user input
    if user_input:
        data = get_bot_response(user_input)
        # Call middleman function with user_input, context, and data
        system_out = middleman(user_input, context, data)
        # Print the system output to console
        print(f"System Output: {system_out}")
        
        # Add user input and system output to context
        context.append({"user": user_input, "assistant": system_out})

def play_tts_audio(text):
    """Play TTS audio for the given text"""
    # Speak out the system output using TTS
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_filename = temp_file.name
    
    save_audio_from_text(text, temp_filename, 0)
    
    # Play the audio
    pygame.mixer.init()
    pygame.mixer.music.load(temp_filename)
    pygame.mixer.music.play()
    
    # Wait for playback to finish, then clean up
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    
    os.unlink(temp_filename)

def get_system_response():
    """Return the current system output"""
    return system_out

def main():
    """Main function to start the transcription app"""
    root = tk.Tk()
    app = TranscriptionApp(root)
    
    # Store reference to update function for potential future use
    app.update_user_data = update_user_data
    app.get_system_response = get_system_response
    app.play_audio_callback = play_tts_audio
    
    root.mainloop()

if __name__ == "__main__":
    main()