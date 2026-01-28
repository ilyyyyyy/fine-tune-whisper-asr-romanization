#Install necessary libraries prior: pip install -q peft transformers datasets accelerate evaluate jiwer

model_name_or_path = "openai/whisper-tiny"
language = "Japanese"
language_abbr = "ja"
task = "transcribe"
dataset_name = "tiny/data.csv"

from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor
from datasets import load_dataset
dataset = load_dataset("csv", data_files=dataset_name)
print(dataset)
