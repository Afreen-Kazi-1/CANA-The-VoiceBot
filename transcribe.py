import asyncio
import pyaudio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, stream, transcript_store, text_widget=None):
        super().__init__(stream)
        self.transcript_store = transcript_store
        self.text_widget = text_widget  # Optional: used in GUI

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                text = result.alternatives[0].transcript
                self.transcript_store["final"] += text + " "
                if self.text_widget:
                    # Update GUI text area from main thread
                    self.text_widget.after(0, lambda: self.text_widget.insert("end", text + "\n"))

async def stream_audio_to_transcribe(stop_event: asyncio.Event, transcript_store, text_widget=None, lang_code="en-US"):
    client = TranscribeStreamingClient(region="us-west-2")

    stream = await client.start_stream_transcription(
        language_code=lang_code,
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    handler = MyEventHandler(stream.output_stream, transcript_store, text_widget)

    audio = pyaudio.PyAudio()
    mic_stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=512,  # Smaller buffer for faster response
    )

    async def mic_to_stream():
        try:
            while not stop_event.is_set():
                data = mic_stream.read(512, exception_on_overflow=False)
                await stream.input_stream.send_audio_event(audio_chunk=data)
        finally:
            await stream.input_stream.end_stream()
            mic_stream.stop_stream()
            mic_stream.close()
            audio.terminate()

    await asyncio.gather(mic_to_stream(), handler.handle_events())

    # Return the final transcript and language code
    return transcript_store.get("final", ""), lang_code
