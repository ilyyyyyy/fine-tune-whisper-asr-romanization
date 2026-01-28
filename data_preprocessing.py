#Install necessary libraries prior: pip install -q peft transformers datasets accelerate evaluate jiwer

model_name_or_path = "openai/whisper-tiny"
language = "Japanese"
language_abbr = "ja"
task = "transcribe"
dataset_name = "tiny/data.csv"

from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor
from datasets import load_dataset, DatasetDict
raw_dataset = load_dataset("csv", data_files=dataset_name)

dataset = raw_dataset["train"].shuffle(seed = 15)
train_test = dataset.train_test_split(test_size=0.2, seed=15)
test_valid = train_test["test"].train_test_split(test_size=0.5, seed=15)

dataset_dict = DatasetDict({
    "train": train_test["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"],
})