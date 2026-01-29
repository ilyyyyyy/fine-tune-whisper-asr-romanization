import evaluate
from transformers import WhisperForConditionalGeneration
from preprocessing import load_and_process_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import gc
from contextlib import nullcontext
from peft import LoraConfig, get_peft_model

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if device.type == "cuda":
    autocast = torch.cuda.amp.autocast()
elif device.type == "mps":
    autocast = torch.autocast(device_type="mps")
else:
    autocast = nullcontext()

dataset_path = "tiny/data.csv"
model_name_or_path = "openai/whisper-tiny"

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.to(device)

dataset_dict, processor, data_collator = load_and_process_data(dataset_path)

forced_decoder_ids = processor.get_decoder_prompt_ids()
tokenizer = processor.tokenizer

metric = evaluate.load("wer")
eval_dataloader = DataLoader(dataset_dict["test"],batch_size=8,collate_fn=data_collator)
model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with autocast:
        with torch.no_grad():
            generated_tokens = model.generate(input_features=batch["input_features"].to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=255).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    del generated_tokens, labels, batch
    gc.collect()
WER = 100 * metric.compute()
print(f"{WER=}")