# cleanAudio.py

import sys
import whisper
from pydub import AudioSegment, silence
import os

input_path = sys.argv[1]
output_path = sys.argv[2]
txt_output_path = output_path.replace(".wav", ".txt")  # Save alongside the audio

# Load and clean audio
audio = AudioSegment.from_file(input_path)
chunks = silence.split_on_silence(audio, min_silence_len=2000, silence_thresh=-40)
clean_audio = sum(chunks)
clean_audio.export(output_path, format="wav")

# Transcribe
model = whisper.load_model("base")
result = model.transcribe(output_path)

# Save to .txt file
with open(txt_output_path, "w") as f:
    f.write("FULL TRANSCRIPTION:\n")
    f.write(result["text"] + "\n\n")
    f.write("SEGMENTS WITH TIMESTAMPS:\n\n")
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        f.write(f"[{start:.2f} → {end:.2f}] {text}\n")

print(f"Transcription with timestamps saved to: {txt_output_path}")
