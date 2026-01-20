from pydub import AudioSegment
import os

textgrid_dir = "tiny/timestamps/"
txt_dir = "tiny/normalized_txts/"
wav_dir = "tiny/wavs/"
slices_dir = "tiny/slices/"
os.makedirs(slices_dir, exist_ok=True)

def get_timestamps(textgrid):
    timestamps = []
    for i, line in enumerate(textgrid):
        if 'name = "phones"' in line:
            break

        if "text =" in line:
            token = line.split("text =")[1].replace("\n", "").replace(" ", "").replace('"', '')
            if not token:
                continue
            min = float(textgrid[i-2].split("=")[1])
            max = float(textgrid[i-1].split("=")[1])

            timestamps.append({
                "token": token,
                "start": min,
                "end": max
            })
    return timestamps

def align(txt, timestamps):
    i = 0
    collected_tokens = []
    for line in txt:
        line = line.strip()
        collected_text = ""
        start = timestamps[i]["start"]

        while i < len(timestamps) and collected_text != line:
            collected_text += timestamps[i]["token"]
            i+=1

        end = timestamps[i-1]["end"]

        collected_tokens.append((start, end))
    return collected_tokens


for tg_filename in os.listdir(textgrid_dir):
    if not tg_filename.endswith(".TextGrid"):
        continue

    audio_num = os.path.splitext(tg_filename)[0]
    tg_path = os.path.join(textgrid_dir, tg_filename)
    txt_path = os.path.join(txt_dir, f"{audio_num}.txt")
    audio_path = os.path.join(wav_dir, f"{audio_num}.wav")

    with open(tg_path) as f:
        textgrid = f.readlines()
    with open(txt_path) as f:
        txt = f.readlines()

    timestamps = get_timestamps(textgrid)
    aligned = align(txt, timestamps)

    audio = AudioSegment.from_wav(audio_path)

    for idx, (start, end) in enumerate(aligned):
        segment = audio[start * 1000:end * 1000]  # milliseconds
        slice_filename = f"{audio_num}_{idx}.wav"
        slice_path = os.path.join(slices_dir, slice_filename)
        segment.export(slice_path, format="wav")

