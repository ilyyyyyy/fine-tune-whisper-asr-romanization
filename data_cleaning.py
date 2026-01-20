import pandas as pd
import re
import os

folders = ('caucasusnohagetaka', 'gongitsune') #names of books

df = pd.read_csv('metadata.csv')
mask = df['wav_path'].str.contains('|'.join(folders))
df = df[mask]
df.reset_index(drop=True, inplace=True)

romaji_df = df.drop(columns=['mp3_path', 'ID1', 'ID2', 'japanese', 'audio_num'])
jp_df = df.drop(columns=['mp3_path', 'ID1', 'ID2', 'romaji', 'audio_num'])

jp_map = {',' : ' ', '.' : ' ', '、' : ' ', '。' : ' ', '「' : ' ', '」' : ' ', '…' : ' ', '？' : ' '}
romaji_map = {'N' : 'n', 'q ' : ' ', 'a:':'aa','i:':'ii','u:':'uu','e:':'ee','o:':'ou', '.': ' ', ',':' '}

def tokenize_jp(text):
    for old, new in jp_map.items():
        text = text.replace(old, new)
    text = text.replace('_', ' ')
    return text

def normalize_jp(text):
    for old, new in jp_map.items():
        text = text.replace(old, new)
    text = text.replace('_', '')
    text = re.sub(r'\s+', '', text)
    return text

def normalize_romaji(text):
    text = text.replace(' ', '').replace('_', ' ').replace('  ', ' ')
    for old, new in romaji_map.items():
        text = text.replace(old, new)
    text = re.sub(r"q([bcdfghjklmnpqrstvwxyz])", r"\1\1", text)
    return text

def write_txts(df, col_name, process_func, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for wav_file in df['wav_path'].unique():
        filtered_df = df[df['wav_path'] == wav_file]
        txt_filename = os.path.basename(wav_file).replace('.wav', '.txt')
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w') as f:
            for line in filtered_df[col_name]:
                f.write(process_func(str(line)) + '\n')

write_txts(jp_df, 'japanese', tokenize_jp, 'tiny/tokenized_txts_test/')
write_txts(jp_df, 'japanese', normalize_jp, 'tiny/normalized_txts_test/')
write_txts(romaji_df, 'romaji', normalize_romaji, 'tiny/romaji_txts_test/')

