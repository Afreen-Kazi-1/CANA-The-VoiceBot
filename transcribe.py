import asyncio
import pyaudio
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from dotenv import load_dotenv
import os
import boto3

# Load environment variables from .env file
load_dotenv()

# Initialize AWS clients using environment variables
transcribe_client = boto3.client('transcribe')
s3_client = boto3.client('s3')


class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        for result in transcript_event.transcript.results:
            if not result.is_partial:
                print(result.alternatives[0].transcript)


async def stream_audio_to_transcribe():
    client = TranscribeStreamingClient(region="us-west-2")

    stream = await client.start_stream_transcription(
        language_code="hi-IN",             # Use "en-US" for English
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    handler = MyEventHandler(stream.output_stream)

    # Setup microphone stream
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
    )

    async def mic_to_stream():
        print("🎙️ Start speaking into your microphone...")
        try:
            while True:
                data = mic_stream.read(1024, exception_on_overflow=False)
                await stream.input_stream.send_audio_event(audio_chunk=data)
        except KeyboardInterrupt:
            print("🛑 Stopping due to keyboard interrupt...")
        finally:
            await stream.input_stream.end_stream()
            mic_stream.stop_stream()
            mic_stream.close()
            audio.terminate()

    await asyncio.gather(mic_to_stream(), handler.handle_events())


# 👇 Graceful shutdown entry point
if __name__ == "__main__":
    try:
        asyncio.run(stream_audio_to_transcribe())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user (Ctrl+C)")
