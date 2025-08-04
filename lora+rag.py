from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from rag_utils import prepare_rag_context

app = Flask(__name__)

# === Paths ===
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = "./tinyllama_lora_output"
PDF_PATH = "local_data/example.pdf"

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH).merge_and_unload()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/", methods=["GET", "POST"])
def index():
    ai_response = None
    user_input = None
    error = None

    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            try:
                # === RAG: Get relevant context from PDF ===
                context = prepare_rag_context(user_input, pdf_path=PDF_PATH)

                # === Prompt ===
                prompt = f"### Instruction:\n{user_input}\n\n### Input:\n{context}\n\n### Response:"
                print(f"\n[DEBUG] Prompt:\n{prompt}")

                inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"[DEBUG] Raw model output:\n{decoded}")

                # Clean the output
                if "### Response:" in decoded:
                    ai_response = decoded.split("### Response:")[-1].strip()
                else:
                    ai_response = decoded.strip()

                print(f"[DEBUG] Final cleaned response:\n{ai_response}")

            except Exception as e:
                error = f"Error during inference: {e}"
        else:
            error = "Please enter a question."

    return render_template("index.html", ai_response=ai_response, user_input=user_input, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
