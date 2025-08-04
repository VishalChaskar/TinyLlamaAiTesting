from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = "./tinyllama_lora_output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer from adapter (most reliable)
tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
base_model.resize_token_embeddings(len(tokenizer))

# Load LoRA and merge
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model = model.merge_and_unload()
model.to(device)
model.eval()

# 🔍 Try a direct prompt
prompt = "### Instruction:\nWhat is artificial intelligence?\n\n### Input:\n\n### Response:"
print(f"\nPrompt:\n{prompt}\n")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

decoded = tokenizer.decode(output[0], skip_special_tokens=False)
print(f"\n[Raw output with special tokens]:\n{decoded}")

# Try to extract
if "### Response:" in decoded:
    response = decoded.split("### Response:")[-1].strip()
else:
    response = decoded.strip()

print(f"\n[Cleaned response]:\n{response}")
