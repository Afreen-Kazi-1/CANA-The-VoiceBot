import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Initialize ElevenLabs client
load_dotenv()
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API"))  # Load API key from .env

# Dummy pipeline output (replace with actual pipeline output)
dummy_pipeline_output = [
    {"response": "हाँ, बिल्कुल! हमारे Lumpsum Lending फीचर के माध्यम से आप एक साथ कई बोर्रोवेर्स को लोन दे सकते हैं। यह आपको बULK में लोन देने की अनुमति देता है, जिससे आपको ज्यादा विकल्प मिलते हैं और जोखिम भी कम होता है।?"}
]

# Configuration
# OUTPUT_DIR = "/content/"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # Your specified voice
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"

# Ensure output directory exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_audio_from_text(text, output_file, index):
    """
    Convert text to speech and save as MP3 using ElevenLabs.
    Args:
        text (str): Text to convert (e.g., pipeline response).
        output_file (str): Path to save MP3 (e.g., /content/response_0.mp3).
        index (int): Index for error reporting.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        audio_stream = elevenlabs.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format=OUTPUT_FORMAT
        )
        with open(output_file, "wb") as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)
        print(f"Audio saved for response {index} to {output_file}")
        return True
    except Exception as e:
        print(f"Error for response {index}: {e}")
        return False

def generate_speech_from_pipeline(pipeline_output):
    """
    Generate and save speech for each response in pipeline output.
    Args:
        pipeline_output (list): List of dicts from nlp_pipeline.py.
    """
    for index, item in enumerate(pipeline_output):
        response_text = item.get("response", "")
        if not response_text:
            print(f"No response text for item {index}, skipping.")
            continue
        output_file = os.path.join(OUTPUT_DIR, f"response_{index}.mp3")
        success = save_audio_from_text(response_text, output_file, index)
        if not success:
            print(f"Failed to save audio for response {index}: {response_text}")

def main():
    # Use dummy data (replace with actual pipeline output)
    pipeline_output = dummy_pipeline_output
    generate_speech_from_pipeline(pipeline_output)

if __name__ == "__main__":
    main()