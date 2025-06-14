import click
import speech_recognition as sr
from asr import WhisperASR
# from utils import log_conversation
import tempfile
import os
import time
import json
# from pydub import AudioSegment

@click.group()
def cli():
    """Matrix Protocol CLI: Voice-to-Text Conversational Assistant"""
    pass

@cli.command()
@click.argument("audio_path", required=False)
@click.option("--microphone", is_flag=True, help="Use microphone input")
def converse(audio_path, microphone):
    """Transcribe audio (and process text)"""
    temp_file = None
    try:
        asr = WhisperASR(model_name="small")
        audio_source = "microphone" if microphone else audio_path if audio_path else "unknown"

        if microphone:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                click.echo("Speak now...")
                audio = recognizer.listen(source, timeout=6)
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with open(temp_file.name, "wb") as f:
                f.write(audio.get_wav_data())
            audio_path = temp_file.name
        else:
            if not audio_path:
                click.echo(click.style("Error: Audio file required unless --microphone is used", fg="red"))
                return
            if not audio_path.endswith((".wav", ".mp3")):
                click.echo(click.style("Error: Only WAV or MP3 files supported", fg="red"))
                return
            if not os.path.exists(audio_path):
                click.echo(click.style(f"Error: {audio_path} not found", fg="red"))
                return
            # If the audio file is too large, raise error
            if os.path.getsize(audio_path) > 10 * 1024 * 1024:
                click.echo(click.style("Error: File too large (max 10MB)", fg="red"))
                return


        # Transcribe audio
        transcript = asr.transcribe(audio_path)
        
        # Clean up temp file if created
        if temp_file:
            try:
                temp_file.close()  # Ensure file handle is closed
                time.sleep(0.1)    # Give OS time to release the handle
                os.remove(temp_file.name)
            except PermissionError:
                click.echo(click.style("Warning: Could not delete temp file (still in use)", fg="yellow"))


        # RAG processing (commented out until Team Member 2 provides RAGProcessor)
        # rag = RAGProcessor()
        # rag_result = rag.process(transcript["text"])
        # required_keys = ["intent", "sentiment", "response"]
        # if not all(key in rag_result for key in required_keys):
        #     click.echo(click.style("Error: Invalid RAG output", fg="red"))
        #     return
        # result = {
        #     "transcribed_text": transcribe_result["text"],
        #     "language": transcribe_result["language"],
        #     **rag_result
        # }
        # log_conversation(audio_source, transcribe_result, rag_result)

        result = transcript
        # log_conversation(audio_source, transcript, {})

        click.echo(click.style(json.dumps(result, indent=2, ensure_ascii=False), fg="green"))
    except Exception as e:
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        click.echo(click.style(f"Error: Processing failed - {str(e)}", fg="red"))

if __name__ == "__main__":
    cli()