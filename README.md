# 🦙 TinyLlama AI Testing

TinyLlamaAiTesting is a local, offline AI-powered assistant built using `llama-cpp-python`. It runs a TinyLlama model on your machine and allows you to interact with documents, analyze files (like `index.html`), and assist with test automation tasks like generating test cases, interpreting logs, and more.

---

## ✨ Features

- Offline AI assistant (no internet required)
- Loads and runs `TinyLlama` locally via `llama-cpp-python`
- Integrates with test automation for:
  - Test case generation
  - Test data generation
  - Log failure analysis
- Can process and understand your local documentation/files

---

## ⚙️ Tech Stack

- Python 3.11+
- llama-cpp-python
- llama-index or langchain (for RAG, optional)
- PyCharm or VSCode
- Local `.gguf` model file (e.g., TinyLlama)

---

## 📁 Project Structure
graphql
Copy
Edit
TinyLlamaAiTesting/
├── llama_runner.py           # Core LLM interaction
├── models/                   # TinyLlama GGUF model files
├── tests/                    # Selenium or Pytest scripts
├── docs/                     # Project documentation like index.html
├── data/                     # JSON, CSV or other sources
├── README.md
└── requirements.txt


## Create a virtual environment:

- bash
- Copy
- Edit
- python -m venv .venv
- .venv\Scripts\activate  # Windows
- Install dependencies:

- bash
- Copy
- Edit
- pip install -r requirements.txt
- Download your TinyLlama .gguf model (e.g., tinyllama-1.1b.gguf) and place it in the project directory under models/.

## 🚀 Usage
- Run the main script:

- bash
- Copy
- Edit
- python llama_runner.py
- Then input your question, e.g.:

- text
- Copy
- Edit
- "Generate 5 test cases for login.html."
- Or use it inside your automation project by importing:

- python
- Copy
- Edit
- from llama_runner import ask_llama
- ask_llama("Analyze this Selenium test log...")


## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TinyLlamaAiTesting.git
   cd TinyLlamaAiTesting


## Example Prompts
- “Generate edge-case test data for email field.”

- “What test cases can be derived from this HTML?”

- “Explain why the test failed with this stack trace…”