from faster_whisper import WhisperModel

model = WhisperModel("turbo", device="cuda", compute_type="float16")

def transcribe(filepath):
    segments, _ = model.transcribe(filepath, word_timestamps=True)
    for segment in segments:
        yield {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (segment.words or [])],
        }