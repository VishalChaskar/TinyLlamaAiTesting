from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from prepare_data import load_and_prepare_dataset
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Training on: {device}")
# Load tokenizer and model
model_path = "./models/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.to(device)

# Load dataset
dataset = load_and_prepare_dataset("transactions_bank.jsonl",tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama_lora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_steps=50,
    logging_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    report_to="none",
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save final adapter
model.save_pretrained("./tinyllama_lora_output")
tokenizer.save_pretrained("./tinyllama_lora_output")
torch.cuda.empty_cache()
print("Successfully trained model")
