import torch
import os
from dotenv import load_dotenv
from faster_whisper.audio import decode_audio
from pyannote.audio import Pipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

load_dotenv()

SAMPLE_RATE = 16000

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HUGGINGFACE_TOKEN"),
)
pipeline.to(torch.device("cuda"))  # ganti "cpu" jika VRAM 6GB kepenuhan (bersaing dengan whisper & ollama)

def diarize(filepath):
    # decode_audio (PyAV) mendukung mp3/m4a/aac/dll, sedangkan sf.read tidak
    audio = decode_audio(filepath, sampling_rate=SAMPLE_RATE)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
    output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    return [(turn.start, turn.end, speaker) for turn, speaker in output.speaker_diarization]

def assign_speaker(seg, turns):
    overlap = {}
    for start, end, spk in turns:
        o = min(seg["end"], end) - max(seg["start"], start)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0) + o
    return max(overlap, key=overlap.get) if overlap else None

def label_transcript(segments, turns):
    names, lines, last = {}, [], None
    for seg in segments:
        spk = assign_speaker(seg, turns) or last or "UNKNOWN"
        if spk not in names:
            names[spk] = f"Speaker {chr(65 + len(names))}"
        text = seg["text"].strip()
        if spk == last:
            lines[-1] += " " + text
        else:
            lines.append(f"[{names[spk]}] {text}")
        last = spk
    return "\n".join(lines)