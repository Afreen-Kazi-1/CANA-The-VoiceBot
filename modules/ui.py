import tkinter as tk
from tkinter import scrolledtext, Frame, ttk
import asyncio
import threading
from dotenv import load_dotenv
import os
import pygame
from transcribe import stream_audio_to_transcribe

load_dotenv()
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

# Initialize pygame mixer for audio playback
pygame.mixer.init()

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lenden AI Chat Assistant")
        self.root.geometry("700x550")  # Reduced from 900x700
        self.root.configure(bg="#f0f2f5")
        
        # Configure modern styling
        self.setup_styles()
        
        # Main container with padding
        main_container = Frame(root, bg="#f0f2f5")
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)  # Reduced padding
        
        # Header
        header_frame = Frame(main_container, bg="#f0f2f5")
        header_frame.pack(fill=tk.X, pady=(0, 15))  # Reduced padding
        
        title_label = tk.Label(
            header_frame, 
            text="Lenden AI Assistant", 
            font=("Segoe UI", 20, "bold"),  # Reduced from 24
            bg="#f0f2f5",
            fg="#1a1a1a"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame, 
            text="Speak in Hindi or English - AI will respond", 
            font=("Segoe UI", 10),  # Reduced from 12
            bg="#f0f2f5",
            fg="#666666"
        )
        subtitle_label.pack(pady=(3, 0))  # Reduced padding
        
        # Chat container with modern styling
        chat_container = Frame(main_container, bg="#ffffff", relief="flat", bd=0)
        chat_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))  # Reduced padding
        
        # Add subtle shadow effect with multiple frames
        shadow_frame = Frame(main_container, bg="#e0e0e0", height=2)
        shadow_frame.pack(fill=tk.X, pady=(0, 13))  # Reduced padding
        
        self.chat_frame = Frame(chat_container, bg="#ffffff", padx=15, pady=15)  # Reduced padding
        self.chat_frame.pack(fill=tk.BOTH, expand=True)
        
        # Chat display area with modern styling
        self.chat_area = scrolledtext.ScrolledText(
            self.chat_frame, 
            wrap=tk.WORD, 
            width=60,  # Reduced from 70
            height=16,  # Reduced from 22
            font=("Segoe UI", 10),  # Reduced from 11
            bg="#fafafa",
            fg="#2c2c2c",
            relief="flat",
            bd=0,
            padx=12,  # Reduced padding
            pady=12,  # Reduced padding
            selectbackground="#e3f2fd",
            insertbackground="#1976d2"
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        
        # Configure chat message styling
        self.chat_area.tag_configure("user", 
                                   background="#e3f2fd", 
                                   foreground="#1565c0",
                                   font=("Segoe UI", 10, "normal"),  # Reduced font size
                                   lmargin1=15, lmargin2=15, rmargin=15,  # Reduced margins
                                   spacing1=6, spacing3=6)  # Reduced spacing
        
        self.chat_area.tag_configure("assistant", 
                                   background="#f1f8e9", 
                                   foreground="#2e7d32",
                                   font=("Segoe UI", 10, "normal"),  # Reduced font size
                                   lmargin1=15, lmargin2=15, rmargin=15,  # Reduced margins
                                   spacing1=6, spacing3=6)  # Reduced spacing
        
        # Control panel with modern buttons
        control_frame = Frame(main_container, bg="#f0f2f5")
        control_frame.pack(fill=tk.X, pady=(0, 8))  # Reduced padding
        
        # Center the buttons
        button_container = Frame(control_frame, bg="#f0f2f5")
        button_container.pack(expand=True)
        
        # Modern button styling
        button_style = {
            "font": ("Segoe UI", 10, "bold"),  # Reduced font size
            "relief": "flat",
            "bd": 0,
            "padx": 20,  # Reduced padding
            "pady": 10,  # Reduced padding
            "cursor": "hand2"
        }
        
        self.hindi_button = tk.Button(
            button_container, 
            text="🎤 Hindi", 
            command=lambda: self.start_transcription("hi-IN"),
            bg="#ff6b35",
            fg="white",
            activebackground="#e55a2b",
            activeforeground="white",
            **button_style
        )
        self.hindi_button.pack(side=tk.LEFT, padx=(0, 12))  # Reduced padding
        
        self.english_button = tk.Button(
            button_container, 
            text="🎤 English", 
            command=lambda: self.start_transcription("en-US"),
            bg="#1976d2",
            fg="white",
            activebackground="#1565c0",
            activeforeground="white",
            **button_style
        )
        self.english_button.pack(side=tk.LEFT, padx=(0, 12))  # Reduced padding
        
        self.stop_button = tk.Button(
            button_container, 
            text="⏹ Stop", 
            command=self.stop_transcription, 
            state=tk.DISABLED,
            bg="#d32f2f",
            fg="white",
            activebackground="#c62828",
            activeforeground="white",
            **button_style
        )
        self.stop_button.pack(side=tk.LEFT)
        
        # Status with modern styling
        self.status_label = tk.Label(
            main_container, 
            text="Ready to listen...", 
            font=("Segoe UI", 9),  # Reduced font size
            bg="#f0f2f5",
            fg="#1976d2",
            pady=8  # Reduced padding
        )
        self.status_label.pack()
        
        # Add welcome message
        self.add_welcome_message()
        
        self.stop_event = None
        self.transcript_store = {"final": ""}
        self.lang_code = "en-US"
        self.thread = None
        self.loop = None
        self.update_user_data = None  # Callback function for storing user data
        self.get_system_response = None  # Callback to get processed response
        self.play_audio_callback = None  # Callback to play TTS audio

    def setup_styles(self):
        """Configure modern UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
    def add_welcome_message(self):
        """Add a welcome message to the chat"""
        welcome_text = "Welcome to Lenden AI Assistant! 👋\n\nClick on Hindi or English button and start speaking. I'll respond with audio playback."
        self.chat_area.insert(tk.END, welcome_text + "\n\n", "assistant")
        self.chat_area.see(tk.END)

    def add_message(self, text, sender):
        # Add spacing between messages
        current_content = self.chat_area.get("1.0", tk.END).strip()
        if current_content:
            self.chat_area.insert(tk.END, "\n")
        
        # Add sender label with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M")
        
        if sender == "user":
            sender_label = f"You ({timestamp})"
            self.chat_area.insert(tk.END, f"{sender_label}\n", "user")
            self.chat_area.insert(tk.END, f"{text}\n\n", "user")
        else:
            sender_label = f"Lenden AI ({timestamp})"
            self.chat_area.insert(tk.END, f"{sender_label}\n", "assistant")
            self.chat_area.insert(tk.END, f"{text}\n\n", "assistant")
        
        self.chat_area.see(tk.END)
        # Force update the display
        self.root.update_idletasks()
        
    def play_audio_response(self):
        """Play audio using the callback if available, otherwise play sample.mp3"""
        try:
            # Update status to show audio is playing
            self.status_label.config(text="🔊 Playing audio response...", fg="#2e7d32")
            self.root.update_idletasks()
            
            # Use callback if available
            if self.play_audio_callback and hasattr(self, 'last_response'):
                self.play_audio_callback(self.last_response)
            else:
                # Fallback to sample.mp3
                pygame.mixer.music.load("sample.mp3")
                pygame.mixer.music.play()
                
                # Wait for the audio to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                    self.root.update_idletasks()
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.status_label.config(text="Audio playback failed", fg="#d32f2f")
        finally:
            # Reset status
            self.status_label.config(text="Ready to listen...", fg="#1976d2")

    async def get_model_response(self, query):
        # Use the system response if callback is available
        if self.get_system_response:
            return self.get_system_response()
        return "Hello world"

    def start_transcription(self, lang_code):
        self.lang_code = lang_code
        lang_name = "Hindi" if lang_code == "hi-IN" else "English"
        self.status_label.config(text=f"🎤 Listening in {lang_name}...", fg="#ff6b35")
        
        # Update button states with visual feedback
        self.stop_button.config(state=tk.NORMAL, bg="#d32f2f")
        self.hindi_button.config(state=tk.DISABLED, bg="#cccccc")
        self.english_button.config(state=tk.DISABLED, bg="#cccccc")
        
        self.transcript_store = {"final": ""}
        self.thread = threading.Thread(target=self.run_async_transcription)
        self.thread.start()

    def stop_transcription(self):
        self.status_label.config(text="⏸ Stopping...", fg="#d32f2f")
        self.stop_button.config(state=tk.DISABLED, bg="#cccccc")
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
            
            # Add user message and update display
            self.root.after(0, lambda: self.add_message(final_text, "user"))
            
            # Get AI response
            response = await self.get_model_response(final_text)
            self.last_response = response  # Store for audio playback
            
            # Add AI response and update display
            self.root.after(0, lambda: self.add_message(response, "assistant"))
            
            # Play audio response after chat is displayed
            self.root.after(500, self.play_audio_response)
        
        # Reset UI state
        self.root.after(0, lambda: self.status_label.config(text="Ready to listen...", fg="#1976d2"))
        self.root.after(0, lambda: self.hindi_button.config(state=tk.NORMAL, bg="#ff6b35"))
        self.root.after(0, lambda: self.english_button.config(state=tk.NORMAL, bg="#1976d2"))

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()