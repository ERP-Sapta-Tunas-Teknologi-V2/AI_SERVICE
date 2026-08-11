import whisper

model = whisper.load_model("turbo")

def transcribe(filepath):
    result = model.transcribe(filepath)
    transcript = result["text"]
    print(transcript)
    return transcript