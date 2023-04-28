import torch
import whisper
import os
import base64
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = whisper.load_model("large-v2")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    if mp3BytesString == None:
        return {'error': "No input provided", "status": 400}
    
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3','wb') as file:
        file.write(mp3Bytes.getbuffer())
    
    # Run the model
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe("input.mp3", **translate_options, word_timestamps=True)
    output = {"output": result, "status": 200}

    os.remove("input.mp3")
    # Return the results as a dictionary
    return output