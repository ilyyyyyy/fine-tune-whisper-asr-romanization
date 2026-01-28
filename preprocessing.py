#Install necessary libraries prior: pip install -q peft transformers datasets accelerate evaluate jiwer torchcodec

model_name_or_path = "openai/whisper-tiny"
language = "Japanese"
language_abbr = "ja"
task = "transcribe"
dataset_name = "tiny/data.csv"

"""Imports! """
from datasets import load_dataset, DatasetDict, Audio
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor
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

def load_and_process_data(
    dataset_path: str,
    model_name_or_path: str = "openai/whisper-tiny",
    audio_column: str = "wav_path",
    transcript_column: str = "romaji",
    language: str = "Japanese",
    task: str = "transcribe",
    seed: int = 15,
    num_proc: int = 2
):
    raw_dataset = load_dataset("csv", data_files=dataset_path)
    dataset = raw_dataset["train"].shuffle(seed=seed)
    train_test = dataset.train_test_split(test_size=0.2, seed=seed)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=seed)
    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    })
    dataset_dict = dataset_dict.cast_column(audio_column, Audio())

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
    processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)

    def prepare_dataset(batch):
        audio = batch[audio_column]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch[transcript_column]).input_ids
        return batch

    dataset_dict = dataset_dict.map(
        prepare_dataset,
        remove_columns=[audio_column, transcript_column],
        num_proc=num_proc
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    return dataset_dict, processor, data_collator
