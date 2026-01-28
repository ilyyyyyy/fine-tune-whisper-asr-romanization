from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import AutoModelForSpeechSeq2Seq
model_name_or_path = "openai/whisper-tiny"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.print_trainable_parameters()