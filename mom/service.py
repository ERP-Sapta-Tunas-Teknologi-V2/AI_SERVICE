from .transcribe import transcribe
from .minutes import generate_minutes

def generate_mom(filepath):
    print("Transcribing...")
    transcript_parts = []
    for segment in transcribe(filepath):
        transcript_parts.append(segment["text"])
        yield {"type": "transcript", "content": segment["text"], "start": segment["start"], "end": segment["end"]}
    transcript = " ".join(transcript_parts)
    yield {"type": "transcript_done", "content": transcript}

    print("Generating MoM...")
    for event in generate_minutes(transcript):
        yield event

    print("Done.")