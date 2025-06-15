# CANA
CANA - Conversational Audio-to-Text Neural Assistant is an emotionally intelligent voice assistant.

We propose a deployable, intelligent voice-to-voice conversational assistant that replicates the fluency and emotional sensitivity of a human sales representative. Designed for practical use in sales, accessibility, and support contexts, the system handles noisy, accented, multilingual, and ambiguous voice inputs while generating context-aware, emotionally attuned responses.

## Table of Contents
* [About the Project]()
* [Process Flow]()
* [Applications]()
* [Folder Structure]()
* [Installations and Execution]()
* [Tech Stack]()
* [Contributors]()
* [Acknowledgements and Resources]()

## About the Project: 
VoiceBot is a multilingual, emotionally aware, and sales-driven conversational AI assistant designed for the P2P lending domain for LenDenClub. It enables users to ask finance-related queries via voice, and the bot responds with relevant, factually grounded information, empathetically, like a human would do — using both text and hyper-realistic speech output.

## Process Flow
Key Focus: Voice Input ➝ Automatic Speech Recognition ➝ Intent and Ambiguity Recognition ➝ RAG based retrieval from Knowledge Base ➝ Text Response Generation ➝ Text + Speech Output

## Applications: 
- P2P Lending Education & Awareness
- Customer Support in Finance
- Voice-Enabled FAQ Assistants
- Sales Assistant for Financial Products
- Low-Literacy Region Financial Onboarding

## Folder Structure
📦VoiceBot_HackstreetGirls_Submission
┣ 📂config                            # Configuration files for API keys and other constants
┃ ┗ 📜config.yaml                     # YAML config file for environment or settings
┣ 📂data                              # Input data for testing and RAG documents
┃ ┣ 📂pdf_dir                         # Folder containing input PDFs
┃ ┗ 📜test.csv                        # Test questions for inference
┣ 📂modules                           # All functional modules of the voice assistant
┃ ┣ 📜asr_module.py                   # Handles Automatic Speech Recognition (ASR)
┃ ┣ 📜chatbot.py                      # LLM interface for generating responses
┃ ┣ 📜intent_recognition.py          # Classifies user intent from input text
┃ ┣ 📜middleman.py                   # Coordinates flow between modules
┃ ┣ 📜transcribe.py                  # Transcribes audio input to text
┃ ┣ 📜tts.py                         # Converts text output into speech
┃ ┣ 📜ui.py                          # Optional UI interface handler
┃ ┗ 📜utils.py                       # Utility functions (embedding, chunking, etc.)
┣ 📂output                            # Stores output CSV responses
┣ 📂rag_cache                         # Cached embeddings and processed text chunks (.pkl files)
┣ 📜.gitignore                        # Files and folders to ignore in version control
┣ 📜cli.py                            # Command-line interface for interaction
┣ 📜main.py                           # Entry point to run the full voice assistant pipeline
┣ 📜README.md                         # Project documentation and instructions
┣ 📜requirements.txt                  # Required Python packages
┣ 📜response.mp3                      # Example output audio response
┣ 📜run_inference.py                 # Script to test RAG responses using test CSV
┗ 📜setup.py                          # Setup script for installation (optional)

## Installations and Execution

Follow these steps to set up and run the VoiceBot project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/VoiceBot_HackstreetGirls_Submission.git
cd VoiceBot_HackstreetGirls_Submission
```

### 2. Place the Test File
Place your test.csv containing the questions in the data/ folder:
```bash
VoiceBot_HackstreetGirls_Submission/
┗ 📂data/
   ┗ 📜test.csv
```

### 3. Create a Virtual Environment and Install Requirements
It's recommended to use a virtual environment to avoid dependency issues:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
### 4. Load Environment Variables
Load the following environment variables in the .env file:
1) GROQ_API_KEY
2) AWS_ACCESS_KEY_ID
3) AWS_SECRET_ACCESS_KEY
4) AWS_REGION
5) AWS_SESSION_TOKEN
6) ELEVENLABS_API

### 5. Run RAG Pipeline Inference fir csv dataset
To generate responses for the test questions using the RAG pipeline:
```bash
python run_inference.py --test_csv ./data/test.csv --output_csv ./output/hackstreetgirls_submission.csv
```
This will read questions from test.csv run the rag pipeline and save the RAG-based answers in output/responses.csv.

### 6. Run the Full Voice Assistant
To run the main conversational voicebot system with voice input/output:

```bash
python main.py
```

This will start the voicebot and allow real-time speech interaction via the microphone and speakers.

## Tech Stack

| Layer                         | Tool / Service                                                |
|-----------------------------  |---------------------------------------------------------------|
| **Speech-to-Text**            | [AWS Transcribe](https://aws.amazon.com/transcribe/)          |
| **Embeddings**                | `transformers`, `LaBSE` / `sentence-transformers`             |
| **Vector Search (RAG)**       | `FAISS`, `numpy`                                              |
| **Voice Output (TTS)**        | `PyAudio`,`ElevenLabs`                                        |
| **NLP & Tokenization**        | `nltk`, `regex`, `torch`, `tokenizers`                        |
| **Data Handling & Utilities** | `pandas`, `argparse`, `dotenv`                                |
| **Models**                    | `Llama 3.3`, `mDeBERTa`, ``,                                  |

## Contributors
*[Afreen Kazi] (https://github.com/Afreen-Kazi-1)
*[Carol Chopde] (https://github.com/CarolChopde)
*[Ghruank Kothare] (https://github.com/Ghruank)
*[Niharika Hariharan] (https://github.com/niharikah005)






