import os
import csv

# Paths
slices_dir = "tiny/slices"
txt_dir = "tiny/romaji_txts"
output_csv = "tiny/data.csv"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["wav_path", "romaji"])

    # Loop through all txts
    for txt_file in os.listdir(txt_dir):

        txt_path = os.path.join(txt_dir, txt_file)
        base_name = os.path.splitext(txt_file)[0]  # e.g. caucasusnohagetaka_01_toyoshima_64kb

        with open(txt_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Map each line to its corresponding WAV
        for i, romaji in enumerate(lines):
            wav_filename = f"{base_name}_{i}.wav"
            wav_path = os.path.join(slices_dir, wav_filename)
            writer.writerow([wav_path, romaji])