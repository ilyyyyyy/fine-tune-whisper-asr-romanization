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

def prepare_dataset(batch):
    audio = batch["wav_path"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["romaji"]).input_ids
    return batch

dataset_dict = dataset_dict.map(
    prepare_dataset,
    remove_columns=["wav_path", "romaji"],
    num_proc=2
)
print(dataset_dict["train"][0].keys())

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input_features
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt"
        )

        # Replace padding with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
