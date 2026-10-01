import os
import sys
import torch
import pyannote.audio.utils.reproducibility as repro
from dotenv import load_dotenv
from faster_whisper.audio import decode_audio
from pyannote.audio import Pipeline

def _keep_tf32(device):
    pass

repro.fix_reproducibility = _keep_tf32
for name, mod in list(sys.modules.items()):
    if name.startswith("pyannote") and hasattr(mod, "fix_reproducibility"):
        mod.fix_reproducibility = _keep_tf32

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

load_dotenv()

SAMPLE_RATE = 16000

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HUGGINGFACE_TOKEN"),
)
pipeline.to(torch.device("cuda"))  # ganti "cpu" jika VRAM 6GB kepenuhan (bersaing dengan whisper & ollama)
print("after:", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

def diarize(filepath):
    # decode_audio (PyAV) mendukung mp3/m4a/aac/dll, sedangkan sf.read tidak
    audio = decode_audio(filepath, sampling_rate=SAMPLE_RATE)
    waveform = torch.from_numpy(audio).unsqueeze(0)

    print("before:", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    print("after:", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)

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
        words = seg.get("words") or [{"start": seg["start"], "end": seg["end"], "word": " " + seg["text"].strip()}]
        for w in words:
            spk = assign_speaker({"start": w["start"], "end": w["end"]}, turns) or last or "UNKNOWN"
            if spk not in names:
                names[spk] = f"Speaker {chr(65 + len(names))}"
            if spk == last:
                lines[-1] += w["word"]
            else:
                lines.append(f"[{names[spk]}] {w['word'].strip()}")
            last = spk
    return "\n".join(lines)