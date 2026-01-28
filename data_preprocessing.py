#Install necessary libraries prior: pip install -q peft transformers datasets accelerate evaluate jiwer torchcodec

model_name_or_path = "openai/whisper-tiny"
language = "Japanese"
language_abbr = "ja"
task = "transcribe"
dataset_name = "tiny/data.csv"

from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor
from datasets import load_dataset, DatasetDict, Audio

raw_dataset = load_dataset("csv", data_files=dataset_name)

dataset = raw_dataset["train"].shuffle(seed = 15)
train_test = dataset.train_test_split(test_size=0.2, seed=15)
test_valid = train_test["test"].train_test_split(test_size=0.5, seed=15)

dataset_dict = DatasetDict({
    "train": train_test["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"],
})
dataset_dict = dataset_dict.cast_column("wav_path", Audio())

from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)
