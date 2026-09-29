import os
import time
from .timing_log import log_time, log_space, log_mom, new_run_id
from .transcribe import transcribe
from .minutes import generate_minutes

def generate_mom(filepath, context="-"):
    new_run_id()
    total_start = time.perf_counter()
    minutes = {"abstract_summary": "", "key_points": "", "action_items": ""}

    try:
        print("Transcribing...")
        transcript_parts = []
        start = time.perf_counter()

        for segment in transcribe(filepath):
            transcript_parts.append(segment["text"].strip())
            yield {"type": "transcript", "content": segment["text"], "start": segment["start"], "end": segment["end"]}

        log_time(f"transcription | {os.path.basename(filepath)}", time.perf_counter() - start)
        transcript = " ".join(transcript_parts)
        yield {"type": "transcript_done", "content": transcript}

        print("Generating MoM...")
        start = time.perf_counter()
        for event in generate_minutes(transcript, context):
            if event["type"] == "mom":
                minutes[event["key"]] += event["content"]
            yield event
        log_time("mom generation total", time.perf_counter() - start)

        print("Done.")

    finally:
        log_mom(os.path.basename(filepath), minutes)
        log_time(f"TOTAL | {os.path.basename(filepath)}", time.perf_counter() - total_start)
        log_space()