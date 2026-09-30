import os
import time
from concurrent.futures import ThreadPoolExecutor
from .logging import log_space, new_run_id, log_time, log_transcript, log_mom
from .transcribe import transcribe
from .minutes import generate_minutes
from .diarize import diarize, label_transcript

def timed_diarize(filepath):
    start = time.perf_counter()
    turns = diarize(filepath)
    log_time(f"diarization | {os.path.basename(filepath)}", time.perf_counter() - start)
    return turns

def generate_mom(filepath, context="-"):
    new_run_id()
    total_start = time.perf_counter()
    minutes = {"abstract_summary": "", "key_points": "", "action_items": ""}

    try:
        print("Transcribing & diarizing...")
        segments = []
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=1) as pool:
            diar_future = pool.submit(timed_diarize, filepath)

            for segment in transcribe(filepath):
                segments.append(segment)
                yield {"type": "transcript", "content": segment["text"], "start": segment["start"], "end": segment["end"]}

            log_time(f"transcription | {os.path.basename(filepath)}", time.perf_counter() - start)
            turns = diar_future.result()  # error di thread akan di-raise di sini

        transcript = label_transcript(segments, turns)
        log_transcript(os.path.basename(filepath), transcript)
        yield {"type": "transcript_done", "content": transcript}

        print("Generating MoM...")
        start = time.perf_counter()
        for event in generate_minutes(transcript, context):
            if event["type"] == "mom":
                minutes[event["key"]] += event["content"]
            yield event

        print("Done.")

    finally:
        log_mom(os.path.basename(filepath), minutes)
        log_time(f"TOTAL | {os.path.basename(filepath)}", time.perf_counter() - total_start)
        log_space()