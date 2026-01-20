import os
import csv
data_dir = "data"
csv_path = "metadata.csv"

metadata_files = [f for f in os.listdir(data_dir) if f.endswith(".metadata.txt")]

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "wav_path",
        "audio_num", "mp3_path", "ID1", "ID2", "japanese", "romaji"
    ])

    for meta_file in metadata_files:
        meta_path = os.path.join(data_dir, meta_file)
        with open(meta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")

                # Fill missing columns with empty string if at the end
                while len(parts) < 6:
                    parts.append("")

                # Map MP3 filename to WAV path (column 1 is filename)
                wav_file = parts[1].replace(".mp3", ".wav")
                wav_path = wav_file

                # Write WAV path + all metadata columns
                writer.writerow([wav_path] + parts)