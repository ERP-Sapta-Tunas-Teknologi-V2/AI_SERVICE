from .transcribe import transcribe
from .minutes import generate_minutes

def generate_mom(filepath):
    print("Transcribing...")
    transcript = transcribe(filepath)

    print("Generating MoM...")
    minutes = generate_minutes(transcript)

    return transcript, minutes