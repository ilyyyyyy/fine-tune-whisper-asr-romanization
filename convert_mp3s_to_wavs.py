import subprocess
import os
mp3_dir = "tiny/mp3s"
wav_dir = "tiny/wavs"
os.makedirs(wav_dir, exist_ok=True)

for filename in os.listdir(mp3_dir):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(mp3_dir, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        # Convert to WAV with FFmpeg
        subprocess.run([
            "ffmpeg",
            "-y",  # overwrite if exists
            "-i", mp3_path,
            "-ar", "16000",  # resample to 16 kHz
            "-ac", "1",  # mono
            wav_path
        ])
        print(f"Converted {mp3_path} to {wav_path}")
