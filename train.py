from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import AutoModelForSpeechSeq2Seq
from preprocessing import load_and_process_data

dataset_path = "tiny/data.csv"
model_name_or_path = "openai/whisper-tiny",

dataset_dict, processor, data_collator = load_and_process_data(dataset_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.print_trainable_parameters()