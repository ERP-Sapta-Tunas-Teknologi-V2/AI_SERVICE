from .transcribe import transcribe
from .minutes import generate_minutes

def generate_mom(filepath):
    print("Transcribing...")
    transcript = transcribe(filepath)
    yield {"type": "transcript", "content": transcript}

    print("Generating MoM...")
    for event in generate_minutes(transcript):
        yield event