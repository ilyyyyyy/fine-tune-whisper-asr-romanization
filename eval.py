import evaluate
from transformers import AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Seq2SeqTrainer, Seq2SeqTrainingArguments
from preprocessing import load_and_process_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import gc
from contextlib import nullcontext
from peft import PeftModel, LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.autocast if device.type == "mps" else lambda: nullcontext()

dataset_path = "tiny/data.csv"
model_name_or_path = "openai/whisper-tiny"

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.to(device)

dataset_dict, processor, data_collator = load_and_process_data(dataset_path)

forced_decoder_ids = processor.get_decoder_prompt_ids()

metric = evaluate.load("wer")
eval_dataloader = DataLoader(dataset_dict["test"],batch_size=8,collate_fn=data_collator)
model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with autocast:
        with torch.no_grad():
            generated_tokens = model.generate(
                input_features=batch["input_features"].to(device),
                decoder_input_ids=forced_decoder_ids,
                max_new_tokens=255,
            )
            .cpu()
            .numpy()