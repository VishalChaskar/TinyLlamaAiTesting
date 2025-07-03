from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load the Hugging Face model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
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
            prompt = f"### Instruction:\n{user_input}\n\n### Input:\n\n### Response:"
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
                ai_response = tokenizer.decode(output[0], skip_special_tokens=True)
                ai_response = ai_response.replace(prompt, "").strip()
            except Exception as e:
                error = f"Error during inference: {e}"
        else:
            error = "Please enter a question."

    return render_template("index.html", ai_response=ai_response, user_input=user_input, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')