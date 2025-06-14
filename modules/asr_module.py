import whisper
import os
import torch

class WhisperASR:
    def __init__(self, model_name="small"):
        """Initialize the Whisper model."""
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        """
        Transcribe an audio file and return text with language.
        Args:
            audio_path (str): Path to the audio file (WAV).
        Returns:
            dict: {"text": str, "language": str}
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Transcribe audio
            result = self.model.transcribe(audio_path, fp16=False)
            output = {
                "text": result["text"].strip(),
                "language": result["language"]
            }
            return output
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")