import os
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, TrainingArguments, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Seq2SeqTrainer
from preprocessing import load_and_process_data
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

dataset_path = "tiny/data.csv"
model_name_or_path = "openai/whisper-tiny"
dataset_dict, processor, data_collator = load_and_process_data(dataset_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)


def print_trainable_params(model, label):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{label}")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {100 * trainable / total:.4f}%\n")

full_ft_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)

print_trainable_params(full_ft_model, "Full fine-tuning")
print_trainable_params(model, "PEFT (LoRA)")

training_args = TrainingArguments(
    output_dir="./results/whisper-tiny-romaji",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    warmup_steps=30,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=8,
    logging_steps=25,
    remove_unused_columns=False,
    label_names=["labels"],
)

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    callbacks=[SavePeftModelCallback()],
)
model.config.use_cache = False
trainer.train()
