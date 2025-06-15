import tkinter as tk
from tkinter import scrolledtext, Frame
import asyncio
import threading
from dotenv import load_dotenv
import os
from transcribe import stream_audio_to_transcribe

load_dotenv()
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chat Assistant")
        
        # Chat container
        self.chat_frame = Frame(root)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chat display area
        self.chat_area = scrolledtext.ScrolledText(
            self.chat_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=20,
            font=("Arial", 10)
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        self.chat_area.tag_configure("user", background="#e0e0e0", justify='right')
        self.chat_area.tag_configure("assistant", background="#e3f2fd", foreground="black", justify='left')
        
        # Button frame
        button_frame = Frame(root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.hindi_button = tk.Button(button_frame, text="🎙️ Hindi", command=lambda: self.start_transcription("hi-IN"))
        self.hindi_button.pack(side=tk.LEFT, padx=5)
        
        self.english_button = tk.Button(button_frame, text="🎙️ English", command=lambda: self.start_transcription("en-US"))
        self.english_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="🛑 Stop", command=self.stop_transcription, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack(pady=5)
        
        self.stop_event = None
        self.transcript_store = {"final": ""}
        self.lang_code = "en-US"
        self.thread = None
        self.loop = None
        self.update_user_data = None  # Callback function for storing user data

    def add_message(self, text, sender):
        self.chat_area.insert(tk.END, "\n\n" if self.chat_area.get("1.0", tk.END).strip() else "")
        self.chat_area.insert(tk.END, f"{text}\n", sender)
        self.chat_area.see(tk.END)  # Auto-scroll to bottom
        
    async def get_model_response(self, query):
        # TODO: Replace this with actual model API call
        # Example:
        # response = await call_your_model_api(query)
        # return response
        return "Hello world"

    def start_transcription(self, lang_code):
        self.lang_code = lang_code
        self.status_label.config(text="Listening...")
        self.stop_button.config(state=tk.NORMAL)
        self.hindi_button.config(state=tk.DISABLED)
        self.english_button.config(state=tk.DISABLED)
        
        self.transcript_store = {"final": ""}
        self.thread = threading.Thread(target=self.run_async_transcription)
        self.thread.start()

    def stop_transcription(self):
        self.status_label.config(text="Stopping...")
        self.stop_button.config(state=tk.DISABLED)
        if self.loop and self.stop_event:
            self.loop.call_soon_threadsafe(self.stop_event.set)

    def run_async_transcription(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.stop_event = asyncio.Event()
        self.loop.run_until_complete(self._transcribe_and_finish())

    async def _transcribe_and_finish(self):
        final_text, lang_code = await stream_audio_to_transcribe(
            self.stop_event, self.transcript_store, None, self.lang_code
        )
        
        if final_text.strip():  # Only process if there's actual text
            # Call the callback to store user input and language
            if self.update_user_data:
                self.update_user_data(final_text, lang_code)
            
            # Add user message
            self.root.after(0, lambda: self.add_message(final_text, "user"))
            
            # Get and add AI response
            response = await self.get_model_response(final_text)
            self.root.after(0, lambda: self.add_message(response, "assistant"))
        
        self.root.after(0, lambda: self.status_label.config(text="Ready"))
        self.root.after(0, lambda: self.hindi_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.english_button.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()