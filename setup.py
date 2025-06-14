from setuptools import setup, find_packages

setup(
    name="CANA-voicebot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click==8.0.1",
        "pydub==0.25.1",
        "speechrecognition==3.8.1",
        "pocketsphinx==0.1.15",
        "openai-whisper==20231117",
        "googletrans==4.0.0-rc1",
        "torch==1.9.0",
        "librosa==0.8.1",
        "soundfile==0.10.3",
    ],
    entry_points={
        "console_scripts": [
            "matrix=matrix.cli:cli",
        ],
    },
    author="Hackstreet Girls",
    description="Matrix Protocol Hackathon: Voice-to-Text Conversational Assistant",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Afreen-Kazi-1/voicebot_hackstreetgirls_subscription",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)