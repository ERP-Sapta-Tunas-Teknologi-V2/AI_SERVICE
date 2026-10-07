import os
import torch
from faster_whisper import WhisperModel

default_device = "cuda" if torch.cuda.is_available() else "cpu"
default_compute_type = "float16" if torch.cuda.is_available() else "int8"

whisper_device = os.getenv("WHISPER_DEVICE", default_device)
whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", default_compute_type)

model = WhisperModel("turbo", device=whisper_device, compute_type=whisper_compute_type)

def transcribe(filepath):
    segments, _ = model.transcribe(filepath, word_timestamps=True)
    for segment in segments:
        yield {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (segment.words or [])],
        }