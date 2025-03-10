# This file runs during container build time to get model weights built into the container

# In this example: An OpenAI Whisper model
import whisper
import torch

def download_model():
    model = whisper.load_model("large-v2")

if __name__ == "__main__":
    download_model()